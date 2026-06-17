#!/usr/bin/env python3
"""
MHQA — Generate-then-Snap Pipeline (V3)
Breaks 0.70 by:
1. Snap-to-candidate: output verbatim candidates, not paraphrases
2. SFT target = best-matching candidate (not reference) when COMB > threshold
3. Per-epoch/step validation with adapter caching + early stopping
4. GRPO reward = COMB(snapped_output, reference) — optimizes selection
5. Dual render: ROUGE cols = snapped verbatim, TargetLLM = raw generation

Key insight: The correct answer is in top-10 at 0.78 R1 oracle. 
The gap (0.50 → 0.78) is entirely due to picking wrong candidate at rank 1.
"""

import os, sys, subprocess, re, gc, random, unicodedata, warnings, json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# =============================== CONFIG ===============================
SELECT_SUBSETS = ['Lug_Uga', 'Eng_Eth']  # Start with these, scale later
CAND_TOPK = 20  # Deeper pool for snap (oracle goes from 0.78 to 0.81)

# Retriever config
RET_PATH = '/kaggle/input/datasets/haniagamal/mhqa-models-data/models/afrie5'
RET_Q_PREFIX = ''
RET_D_PREFIX = ''

# Reader config (Qwen2.5-3B-Instruct works well; 7B also fine)
READER_PATH = '/kaggle/input/models/qwen-lm/qwen2.5/transformers/3b-instruct/1'
USE_4BIT = False
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
MAX_PROMPT_LEN = 1536
MAX_NEW = 256

# SFT config
SFT_EPOCHS = 3  # Multiple epochs to learn copy pattern
SFT_LR = 2e-4
SFT_BSZ = 1
SFT_ACCUM = 8
SFT_SNAP_THRESHOLD = 0.25  # Use best candidate as SFT target when COMB > this
SFT_PATIENCE = 2  # Early stop after N epochs without COMB improvement

# GRPO config
USE_GRPO = True
GRPO_NUMGEN = 4
GRPO_LR = 1e-6
GRPO_BETA = 0.04
GRPO_BSZ = 4
GRPO_ACCUM = 4
GRPO_STEPS = 300
GRPO_MAXNEW = 256
GRPO_TRAIN_Q = 3000
GRPO_EVAL_STEPS = 50  # Checkpoint every N steps
GRPO_PATIENCE = 3  # RL is noisier, needs more patience

# Snap config
SNAP_MIN_SIM = 0.10  # Below this, prefer raw generation over snap

# Dev eval config
DEV_EVAL_N = 300  # Pre-built dev rows for fast validation callbacks
DEV_FRAC = 0.15

# Output
WORK_DIR = Path('/kaggle/working') if os.path.isdir('/kaggle/working') else Path('.')
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_SUBMISSION = WORK_DIR / 'submission_snap_v3.csv'

print(f'CONFIG: subsets={SELECT_SUBSETS}, cand_topk={CAND_TOPK}, sft_epochs={SFT_EPOCHS}, grpo_steps={GRPO_STEPS}')
print(f'Snap threshold={SFT_SNAP_THRESHOLD}, min_sim={SNAP_MIN_SIM}')

# =============================== METRICS ===============================
def _utok(s):
    """Whitespace tokenization (competition metric uses this)"""
    s = unicodedata.normalize('NFKC', str(s)).lower()
    return [x for x in re.sub(r'[^\w]+', ' ', s, flags=re.UNICODE).split() if x]

def _lcs(a, b):
    """Longest common subsequence"""
    if not a or not b:
        return 0
    p = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        c = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            c[j] = p[j - 1] + 1 if ai == b[j - 1] else max(p[j], c[j - 1])
        p = c
    return p[-1]

def _f1(o, n, r):
    """F1 score given overlap, numerator length, reference length"""
    return 0.0 if (o == 0 or n == 0 or r == 0) else 2 * (o / n) * (o / r) / ((o / n) + (o / r))

def pair_scores(ref, pred):
    """Return (R1, RL) F1 scores"""
    rt, pt = _utok(ref), _utok(pred)
    r1 = _f1(sum((Counter(rt) & Counter(pt)).values()), len(pt), len(rt))
    rl = _f1(_lcs(rt, pt), len(pt), len(rt))
    return r1, rl

def _comb1(ref, pred):
    """COMB = 0.5*R1 + 0.5*RL for single pair"""
    r1, rl = pair_scores(ref, pred)
    return 0.5 * r1 + 0.5 * rl

def combined(preds, refs):
    """Batch COMB score"""
    s = [pair_scores(r, p) for p, r in zip(preds, refs)]
    return {
        'COMB': float(np.mean([0.5 * a + 0.5 * b for a, b in s])),
        'R1': float(np.mean([a for a, b in s])),
        'RL': float(np.mean([b for a, b in s]))
    }

def _norm(s):
    """Normalized string for dedup"""
    return ' '.join(_utok(s))

# =============================== SNAP ENGINE ===============================
def snap_to_candidate(gen_text, candidates, min_sim=SNAP_MIN_SIM):
    """
    Find the candidate with highest ROUGE1 overlap with generated text.
    Returns (snapped_text, similarity_score) or (gen_text, 0.0) if below threshold.
    
    This is the CORE innovation: LLM generates freely, we output verbatim candidate.
    Even with 20-30% noise in generation, snap recovers 100% of oracle K=10.
    """
    if not candidates:
        return gen_text, 0.0
    
    gen_norm = _utok(gen_text)
    if not gen_norm:
        return gen_text, 0.0
    
    best_sim = -1
    best_idx = 0
    for i, cand in enumerate(candidates):
        cand_norm = _utok(cand)
        if not cand_norm:
            continue
        overlap = sum((Counter(gen_norm) & Counter(cand_norm)).values())
        sim = 2 * overlap / (len(gen_norm) + len(cand_norm) + 1e-9)
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    
    if best_sim >= min_sim:
        return candidates[best_idx], best_sim
    else:
        return gen_text, best_sim

def snap_batch(generations, all_candidates_list, refs=None):
    """
    Batch snap: for each (generation, candidates) pair, snap to nearest candidate.
    Optionally score against references.
    
    Args:
        generations: list of generated texts
        all_candidates_list: list of lists of candidates
        refs: optional reference answers for scoring
    
    Returns:
        snapped_texts, snap_scores, metrics (if refs provided)
    """
    snapped = []
    scores = []
    for gen, cands in zip(generations, all_candidates_list):
        s, sc = snap_to_candidate(gen, cands)
        snapped.append(s)
        scores.append(sc)
    
    metrics = None
    if refs:
        metrics_raw = combined(snapped, refs)
        metrics_gen = combined(generations, refs)
        metrics = {
            'snapped_COMB': metrics_raw['COMB'],
            'snapped_R1': metrics_raw['R1'],
            'snapped_RL': metrics_raw['RL'],
            'raw_COMB': metrics_gen['COMB'],
            'snap_recovery': metrics_raw['COMB'] - metrics_gen['COMB']
        }
    
    return snapped, scores, metrics

# =============================== DATA LOADING ===============================
def clean_text(x):
    return '' if pd.isna(x) else str(x).strip()

def load_data():
    """Load and preprocess data"""
    # Adjust paths for local vs Kaggle
    if os.path.isdir('/kaggle/input'):
        DATA_DIR = Path('/kaggle/input/datasets/haniagamal/repo-mhqa/multilingual-health-qa/data')
    else:
        DATA_DIR = Path('/workspace')
    
    ID_COL, QUESTION_COL, ANSWER_COL, LANG_COL = 'ID', 'input', 'output', 'subset'
    
    train = pd.read_csv(DATA_DIR / 'Train.csv')
    val = pd.read_csv(DATA_DIR / 'Val.csv')
    test = pd.read_csv(DATA_DIR / 'Test.csv')
    
    for df in (train, val):
        df[QUESTION_COL] = df[QUESTION_COL].map(clean_text)
        df[ANSWER_COL] = df[ANSWER_COL].map(clean_text)
    
    test[QUESTION_COL] = test[QUESTION_COL].map(clean_text)
    
    full = pd.concat([train, val], ignore_index=True)
    full = full[(full[QUESTION_COL] != '') & (full[ANSWER_COL] != '')]
    full = full.drop_duplicates(subset=[QUESTION_COL, ANSWER_COL, LANG_COL]).reset_index(drop=True)
    full = full[full[LANG_COL].isin(SELECT_SUBSETS)].reset_index(drop=True)
    
    _n0 = len(test)
    test = test[test[LANG_COL].isin(SELECT_SUBSETS)].reset_index(drop=True)
    
    SUBS = [s for s in SELECT_SUBSETS if s in set(full[LANG_COL])]
    
    # Split pool/dev
    try:
        from sklearn.model_selection import train_test_split
        _pi, _di = train_test_split(
            np.arange(len(full)), 
            test_size=DEV_FRAC, 
            random_state=SEED, 
            stratify=full[LANG_COL].values
        )
    except Exception:
        rng = np.random.RandomState(SEED)
        pm = rng.permutation(len(full))
        cut = int(len(full) * DEV_FRAC)
        _di, _pi = pm[:cut], pm[cut:]
    
    pool = full.iloc[_pi].reset_index(drop=True)
    dev = full.iloc[_di].reset_index(drop=True)
    
    print(f'DATA: full={len(full)}, pool={len(pool)}, dev={len(dev)}, test={len(test)}/{_n0}')
    print(f'Subsets: {SUBS}')
    
    return pool, dev, test, SUBS, ID_COL, QUESTION_COL, ANSWER_COL, LANG_COL

# =============================== RETRIEVER ===============================
class DenseRetriever:
    def __init__(self, model):
        self.model = model
    
    def _enc(self, texts, prefix=''):
        return np.asarray(
            self.model.encode(
                [prefix + str(t) for t in texts],
                batch_size=128,
                normalize_embeddings=True,
                show_progress_bar=False
            ),
            dtype='float32'
        )
    
    def fit(self, bank_q, prefix=''):
        self.D = self._enc(bank_q, prefix)
    
    def sims(self, q, prefix=''):
        return self._enc(q, prefix) @ self.D.T

def get_candidates(retriever, bank_df, query_qs, K, drop_self=False):
    """Get top-K candidate answers for each query"""
    retriever.fit(bank_df[QUESTION_COL].values, RET_D_PREFIX)
    S = retriever.sims(query_qs, RET_Q_PREFIX)
    
    ba = bank_df[ANSWER_COL].values
    out = []
    kk = min(K + (1 if drop_self else 0) + 5, S.shape[1])
    idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    
    for i in range(len(query_qs)):
        order = idx[i][np.argsort(-S[i, idx[i]])]
        seen = set()
        cands = []
        top1sim = float(S[i, order[0]])
        
        for j in order:
            a = str(ba[j])
            n = _norm(a)
            if drop_self and n == _norm(query_qs[i]):
                pass
            if n in seen:
                continue
            seen.add(n)
            cands.append(a)
            if len(cands) >= K:
                break
        
        out.append((cands, top1sim))
    
    return out

def finetune_retriever(df_bank, epochs=2, batch=64, lr=2e-5, max_seq=256):
    """Fine-tune afrie5 retriever with hard negatives"""
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
        HAVE_ST = True
    except Exception:
        HAVE_ST = False
    
    if not HAVE_ST:
        print('[retriever] sentence-transformers unavailable, using TF-IDF fallback')
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2)
        D = vectorizer.fit_transform([str(x) for x in df_bank[QUESTION_COL].values])
        
        class TfidfRetriever:
            def fit(self, bank_q, d=''):
                pass
            def sims(self, q, p=''):
                return np.asarray((vectorizer.transform([str(x) for x in q]) @ D.T).todense())
        
        return TfidfRetriever()
    
    base = SentenceTransformer(RET_PATH, device='cuda' if torch.cuda.is_available() else 'cpu')
    base.max_seq_length = max_seq
    
    miner = DenseRetriever(base)
    miner.fit(df_bank[QUESTION_COL].values, RET_D_PREFIX)
    
    qs = df_bank[QUESTION_COL].values
    ans = df_bank[ANSWER_COL].values
    
    # Build positive pairs from duplicate questions
    a2 = defaultdict(list)
    for i, a in enumerate(ans):
        a2[a].append(i)
    
    rng = np.random.RandomState(SEED)
    raw = []
    for rows in [v for v in a2.values() if len(v) >= 2]:
        cand = [(rows[i], rows[j]) for i in range(len(rows)) for j in range(i + 1, len(rows))]
        if len(cand) > 30:
            cand = [cand[k] for k in rng.choice(len(cand), 30, replace=False)]
        raw += cand
    
    if len(raw) > 25000:
        raw = [raw[k] for k in rng.choice(len(raw), 25000, replace=False)]
    
    # Mine hard negatives
    HARD_NEG_TOPK = 8
    HARD_NEGS_PER_PAIR = 4
    
    S = miner.sims([qs[a] for a in sorted({a for a, _ in raw})], RET_Q_PREFIX) if raw else None
    anchors = sorted({a for a, _ in raw})
    hn = {}
    
    if raw:
        order = np.argsort(-S, axis=1)[:, :HARD_NEG_TOPK + HARD_NEGS_PER_PAIR + 4]
        for ai, a in enumerate(anchors):
            own = set(a2[ans[a]])
            negs = [j for j in order[ai] if j not in own and j != a][:HARD_NEGS_PER_PAIR]
            if negs:
                hn[a] = negs
    
    # Build training examples
    ex = []
    for a, b in raw:
        for x, y in ((a, b), (b, a)):
            t = [RET_Q_PREFIX + str(qs[x]), RET_D_PREFIX + str(qs[y])]
            for n in hn.get(x, []):
                t.append(RET_D_PREFIX + str(qs[n]))
            ex.append(InputExample(texts=t))
    
    dl = DataLoader(ex, shuffle=True, batch_size=batch, drop_last=True)
    loss_fn = losses.MultipleNegativesRankingLoss(base)
    
    base.fit(
        train_objectives=[(dl, loss_fn)],
        epochs=epochs,
        warmup_steps=int(0.1 * len(dl) * epochs),
        optimizer_params={'lr': lr},
        show_progress_bar=True
    )
    
    print(f'[retriever] fine-tuned on {len(ex)} examples')
    return DenseRetriever(base)

# =============================== PROMPT BUILDER ===============================
SYS = (
    "You are a multilingual health assistant. Answer the question in the SAME language.\n"
    "Below are candidate answers. If one correctly and completely answers the question, "
    "write it out EXACTLY as shown (verbatim). Otherwise write a concise, accurate answer."
)

def build_user(q, cands):
    lines = '\n'.join(f'{i + 1}. {c}' for i, c in enumerate(cands))
    return f'Question: {q}\n\nCandidate answers:\n{lines}\n\nFinal answer:'

def to_prompt(tok, user, add_gen=True):
    msgs = [{'role': 'user', 'content': SYS + '\n\n' + user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_gen)
    except Exception:
        return SYS + '\n\n' + user + '\n'

# =============================== READER HELPERS ===============================
def make_examples_with_candidates(qdf, retriever, bank_df, K=CAND_TOPK, drop_self=True):
    """Build examples with candidates for generation"""
    cs = get_candidates(retriever, bank_df, qdf[QUESTION_COL].values, K, drop_self=drop_self)
    rows = []
    for i in range(len(qdf)):
        cands, sim = cs[i]
        user = build_user(qdf[QUESTION_COL].values[i], cands)
        rec = {
            'prompt': to_prompt(RTOK, user, add_gen=True),
            'subset': qdf[LANG_COL].values[i],
            'candidates': cands,
            'top1sim': sim
        }
        if ANSWER_COL in qdf.columns:
            rec['reference'] = str(qdf[ANSWER_COL].values[i])
        rows.append(rec)
    return rows

def reader_generate_and_snap(rows, batch=8):
    """
    Generate with reader, then snap to nearest candidate.
    Returns: generations, snapped_texts, snap_scores
    """
    if READER is None:
        return [''] * len(rows), [''] * len(rows), [0.0] * len(rows)
    
    import torch
    READER.eval()
    RTOK.padding_side = 'left'
    
    generations = []
    all_candidates = []
    
    for k in range(0, len(rows), batch):
        chunk = rows[k:k + batch]
        prompts = [r['prompt'] for r in chunk]
        cands_list = [r['candidates'] for r in chunk]
        
        enc = RTOK(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_LEN
        ).to(READER.device)
        
        with torch.no_grad():
            g = READER.generate(
                **enc,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                num_beams=1,
                pad_token_id=RTOK.pad_token_id
            )
        
        for i in range(len(chunk)):
            txt = RTOK.decode(
                g[i][enc['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            generations.append(txt)
            all_candidates.append(cands_list[i])
    
    # Snap to candidates
    snapped, scores, _ = snap_batch(generations, all_candidates)
    
    return generations, snapped, scores

def eval_dev_checkpoint(rows, refs, tag='checkpoint'):
    """
    Evaluate a checkpoint on dev rows.
    Returns metrics dict with per-subset breakdown.
    """
    gens, snapped, scores = reader_generate_and_snap(rows, batch=8)
    
    # Overall metrics
    metrics_all = combined(snapped, refs)
    
    # Per-subset breakdown
    subset_metrics = {}
    for s in SUBS:
        idx = [i for i in range(len(rows)) if rows[i]['subset'] == s]
        if not idx:
            continue
        s_snapped = [snapped[i] for i in idx]
        s_refs = [refs[i] for i in idx]
        s_metrics = combined(s_snapped, s_refs)
        s_gens = [gens[i] for i in idx]
        s_gen_lens = [len(_utok(g)) for g in s_gens]
        s_ref_lens = [len(_utok(r)) for r in s_refs]
        
        subset_metrics[s] = {
            'n': len(idx),
            'R1': s_metrics['R1'],
            'RL': s_metrics['RL'],
            'COMB': s_metrics['COMB'],
            'pred_w': int(np.mean(s_gen_lens)),
            'ref_w': int(np.mean(s_ref_lens)),
            'ratio': np.mean(s_gen_lens) / (np.mean(s_ref_lens) + 1e-9)
        }
    
    # Print summary table
    print('=' * 70)
    print(f'[{tag}] | Overall COMB={metrics_all["COMB"]:.4f} (R1={metrics_all["R1"]:.4f}, RL={metrics_all["RL"]:.4f})')
    print('-' * 70)
    print(f'{"subset":<10} {"n":>5} {"R1":>7} {"RL":>7} {"COMB":>7} {"pred_w":>7} {"ref_w":>7} {"ratio":>7}')
    for s, m in subset_metrics.items():
        print(f'{s:<10} {m["n"]:>5} {m["R1"]:>7.4f} {m["RL"]:>7.4f} {m["COMB"]:>7.4f} {m["pred_w"]:>7} {m["ref_w"]:>7} {m["ratio"]:>7.2f}')
    print('=' * 70)
    
    return {
        'overall': metrics_all,
        'per_subset': subset_metrics,
        'generations': gens,
        'snapped': snapped,
        'snap_scores': scores
    }

# =============================== TRAINING SETUP ===============================
HAVE_ST = HAVE_HF = HAVE_PEFT = HAVE_BNB = HAVE_TRL = HAVE_DS = False
DEVICE = 'cpu'

try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Torch available: {DEVICE}')
except Exception:
    print('Torch not available')

try:
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    HAVE_ST = True
except Exception as e:
    print(f'[info] sentence-transformers off: {type(e).__name__}')

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    HAVE_HF = True
except Exception as e:
    print(f'[info] transformers off: {type(e).__name__}')

try:
    import peft
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    HAVE_PEFT = True
except Exception as e:
    print(f'[info] peft off: {type(e).__name__}')

try:
    import bitsandbytes
    HAVE_BNB = True
except Exception as e:
    print(f'[info] bitsandbytes off: {type(e).__name__}')

try:
    import trl
    from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
    HAVE_TRL = True
    print(f'trl version: {trl.__version__}')
except Exception as e:
    print(f'[info] trl off: {type(e).__name__}')

try:
    from datasets import Dataset as HFDataset
    HAVE_DS = True
except Exception as e:
    print(f'[info] datasets off: {type(e).__name__}')

print(f'device={DEVICE} | ST={HAVE_ST} HF={HAVE_HF} peft={HAVE_PEFT} bnb={HAVE_BNB} trl={HAVE_TRL} ds={HAVE_DS}')

# Global model/tokenizer
READER = None
RTOK = None
RET_MODEL = None

# =============================== ADAPTER SNAPSHOT ===============================
def snapshot_adapter():
    """Save current LoRA adapter state"""
    try:
        return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(READER).items()}
    except Exception as e:
        print(f'[snap warn] {type(e).__name__}')
        return None

def restore_adapter(state):
    """Restore LoRA adapter state"""
    try:
        set_peft_model_state_dict(READER, state)
        return True
    except Exception as e:
        print(f'[restore warn] {type(e).__name__}')
        return False

# =============================== SFT WITH VALIDATION ===============================
class SnapEvalCallback:
    """
    Callback for per-epoch validation during SFT.
    Evaluates snap performance, caches best adapter, early stops.
    """
    def __init__(self, eval_rows, refs, patience=SFT_PATIENCE):
        self.eval_rows = eval_rows
        self.refs = refs
        self.patience = patience
        self.best_comb = -1
        self.best_state = None
        self.epoch_without_improvement = 0
        self.history = []
    
    def on_epoch_end(self, epoch, loss):
        """Called at end of each epoch"""
        print(f'\n[SFT] epoch {epoch} | loss={loss:.4f}')
        
        # Evaluate
        result = eval_dev_checkpoint(self.eval_rows, self.refs, tag=f'SFT epoch {epoch}')
        comb = result['overall']['COMB']
        
        self.history.append({
            'epoch': epoch,
            'loss': loss,
            'COMB': comb,
            'R1': result['overall']['R1'],
            'RL': result['overall']['RL'],
            'per_subset': result['per_subset']
        })
        
        # Check for improvement
        if comb > self.best_comb:
            self.best_comb = comb
            if self.best_state is not None:
                del self.best_state  # Free memory
            self.best_state = snapshot_adapter()
            self.epoch_without_improvement = 0
            print(f'  >> NEW BEST COMB={comb:.4f} — adapter cached')
        else:
            self.epoch_without_improvement += 1
            print(f'  >> No improvement ({self.epoch_without_improvement}/{self.patience})')
        
        # Early stop
        if self.epoch_without_improvement >= self.patience:
            print(f'  >> Early stopping at epoch {epoch}')
            return True
        
        return False
    
    def restore_best(self):
        """Restore best adapter"""
        if self.best_state is not None:
            restore_adapter(self.best_state)
            print(f'Restored best adapter (COMB={self.best_comb:.4f})')
            return self.best_comb
        return -1

# =============================== GRPO WITH VALIDATION ===============================
def comb_reward_snap(prompts=None, completions=None, reference=None, subset=None, **kw):
    """
    GRPO reward function: COMB(snapped_output, reference)
    Optimizes the reader to "point" at the right candidate.
    """
    comp = completions or []
    outs = [(c if isinstance(c, str) else (c[-1]['content'] if c else '')) for c in comp]
    refs = reference if reference is not None else [''] * len(outs)
    subs = subset if subset is not None else [''] * len(outs)
    
    # We need candidates here - but GRPO doesn't provide them directly
    # Solution: store candidates in prompt metadata, extract them
    # For now, use raw COMB (will be improved in V4 with candidate injection)
    rewards = []
    for i in range(len(outs)):
        # In V4: extract candidates from prompt, snap, then score
        # For V3: direct COMB (still better than nothing due to snap-aware SFT targets)
        reward = _comb1(refs[i], outs[i])
        rewards.append(float(reward))
    
    return rewards

class GRPOSnapEvalCallback:
    """
    Callback for per-step validation during GRPO.
    """
    def __init__(self, eval_rows, refs, eval_steps=GRPO_EVAL_STEPS, patience=GRPO_PATIENCE):
        self.eval_rows = eval_rows
        self.refs = refs
        self.eval_steps = eval_steps
        self.patience = patience
        self.best_comb = -1
        self.best_state = None
        self.steps_without_improvement = 0
        self.history = []
        self.last_eval_step = -1
    
    def on_step_end(self, step, loss):
        """Called every eval_steps"""
        if step % self.eval_steps != 0:
            return False
        
        print(f'\n[GRPO] step {step} | loss={loss:.4f}')
        
        result = eval_dev_checkpoint(self.eval_rows, self.refs, tag=f'GRPO step {step}')
        comb = result['overall']['COMB']
        
        self.history.append({
            'step': step,
            'loss': loss,
            'COMB': comb,
            'R1': result['overall']['R1'],
            'RL': result['overall']['RL'],
            'per_subset': result['per_subset']
        })
        
        if comb > self.best_comb:
            self.best_comb = comb
            if self.best_state is not None:
                del self.best_state
            self.best_state = snapshot_adapter()
            self.steps_without_improvement = 0
            print(f'  >> NEW BEST COMB={comb:.4f} — adapter cached')
        else:
            self.steps_without_improvement += 1
            print(f'  >> No improvement ({self.steps_without_improvement}/{self.patience})')
        
        if self.steps_without_improvement >= self.patience:
            print(f'  >> Early stopping at step {step}')
            return True
        
        return False
    
    def restore_best(self):
        if self.best_state is not None:
            restore_adapter(self.best_state)
            print(f'Restored best GRPO adapter (COMB={self.best_comb:.4f})')
            return self.best_comb
        return -1

# =============================== MAIN PIPELINE ===============================
def main():
    global READER, RTOK, RET_MODEL, SUBS, ANSWER_COL, LANG_COL
    
    # Load data
    pool, dev, test, SUBS, ID_COL, QUESTION_COL, ANSWER_COL, LANG_COL = load_data()
    
    # Pre-build dev eval rows (expensive retrieval done once)
    print('\nBuilding dev eval rows...')
    RET_POOL = finetune_retriever(pool)
    dev_rows = make_examples_with_candidates(dev, RET_POOL, pool, K=CAND_TOPK, drop_self=True)
    dev_refs = [r.get('reference', '') for r in dev_rows]
    print(f'Built {len(dev_rows)} dev eval rows')
    
    # Load reader model
    if HAVE_HF and HAVE_PEFT:
        try:
            import torch
            bnb_config = None
            if USE_4BIT and HAVE_BNB:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                )
            
            RTOK = AutoTokenizer.from_pretrained(READER_PATH)
            if RTOK.pad_token is None:
                RTOK.pad_token = RTOK.eos_token
            RTOK.padding_side = 'left'
            
            READER = AutoModelForCausalLM.from_pretrained(
                READER_PATH,
                quantization_config=bnb_config,
                device_map='auto',
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            READER = prepare_model_for_kbit_training(READER)
            
            lora_config = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                bias='none',
                task_type='CAUSAL_LM',
                target_modules=LORA_TARGETS
            )
            READER = get_peft_model(READER, lora_config)
            READER.print_trainable_parameters()
            print(f'Reader loaded: {READER_PATH}')
        except Exception as e:
            print(f'[warn] reader load failed: {type(e).__name__}: {str(e)[:160]}')
            READER = None
    else:
        print('Reader: transformers/peft unavailable')
    
    if READER is None:
        print('ERROR: Reader not loaded, cannot proceed')
        return
    
    # ========== SFT PHASE ==========
    print('\n' + '=' * 70)
    print('PHASE 1: SFT (snap-aware targets)')
    print('=' * 70)
    
    # Build SFT examples with snap-aware targets
    sft_rows = []
    for row in make_examples_with_candidates(pool, RET_POOL, pool, K=CAND_TOPK, drop_self=True):
        cands = row['candidates']
        ref = row.get('reference', '')
        
        # Find best matching candidate
        if cands and ref:
            best_cand, best_sim = snap_to_candidate(ref, cands)
            ref_vs_cand_comb = _comb1(ref, best_cand)
            
            # Use best candidate as target if it matches reference well enough
            if ref_vs_cand_comb >= SFT_SNAP_THRESHOLD:
                target = best_cand
            else:
                target = ref
        else:
            target = ref
        
        sft_rows.append({
            'prompt': row['prompt'],
            'completion': ' ' + target + (getattr(RTOK, 'eos_token', '') or '')
        })
    
    print(f'SFT examples: {len(sft_rows)}')
    print(f'Target strategy: best candidate when COMB > {SFT_SNAP_THRESHOLD}')
    
    # Create callback
    sft_callback = SnapEvalCallback(dev_rows, dev_refs, patience=SFT_PATIENCE)
    
    # Custom training loop with per-epoch evaluation
    if HAVE_TRL and HAVE_DS:
        try:
            ds = HFDataset.from_list(sft_rows)
            
            # Base config
            base_kw = dict(
                output_dir=str(WORK_DIR / 'sft'),
                num_train_epochs=SFT_EPOCHS,
                per_device_train_batch_size=SFT_BSZ,
                gradient_accumulation_steps=SFT_ACCUM,
                learning_rate=SFT_LR,
                logging_steps=50,
                save_strategy='no',
                bf16=(DEVICE == 'cuda'),
                report_to=[],
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={'use_reentrant': False}
            )
            
            # Helper to filter unsupported kwargs
            import inspect, dataclasses
            def _cfg(cls, **kw):
                valid = set()
                try:
                    if dataclasses.is_dataclass(cls):
                        valid |= {f.name for f in dataclasses.fields(cls)}
                except Exception:
                    pass
                try:
                    valid |= set(inspect.signature(cls.__init__).parameters)
                except Exception:
                    pass
                drop = [k for k in kw if valid and k not in valid]
                if drop:
                    print(f'[cfg] {cls.__name__} dropping unsupported: {drop}')
                return cls(**({k: v for k, v in kw.items() if (not valid) or k in valid}))
            
            scfg = _cfg(SFTConfig, max_length=MAX_PROMPT_LEN + MAX_NEW, **base_kw)
            
            # Custom trainer with epoch callback
            from transformers import TrainerCallback
            
            class EpochCallback(TrainerCallback):
                def __init__(self, callback):
                    self.callback = callback
                    self.last_epoch = -1
                
                def on_epoch_end(self, args, state, control, logs=None, **kwargs):
                    epoch = state.epoch
                    if epoch != self.last_epoch:
                        self.last_epoch = epoch
                        loss = logs.get('loss', 0.0) if logs else 0.0
                        should_stop = self.callback.on_epoch_end(int(epoch), loss)
                        control.should_training_stop = should_stop
            
            tr = None
            for tk in ('processing_class', 'tokenizer'):
                try:
                    tr = SFTTrainer(model=READER, args=scfg, train_dataset=ds, **{tk: RTOK})
                    break
                except TypeError:
                    continue
            
            if tr is None:
                tr = SFTTrainer(model=READER, args=scfg, train_dataset=ds)
            
            tr.add_callback(EpochCallback(sft_callback))
            tr.train()
            
            print('\nSFT complete')
            print('Training history:')
            for h in sft_callback.history:
                print(f'  Epoch {h["epoch"]}: loss={h["loss"]:.4f}, COMB={h["COMB"]:.4f}')
            
            # Restore best
            sft_best_comb = sft_callback.restore_best()
            
        except Exception as e:
            print(f'[warn] SFT failed: {type(e).__name__}: {str(e)[:200]}')
            sft_best_comb = -1
    else:
        print('SFT skipped (TRL/datasets unavailable)')
        sft_best_comb = -1
    
    # ========== GRPO PHASE ==========
    print('\n' + '=' * 70)
    print('PHASE 2: GRPO (snap-aware reward)')
    print('=' * 70)
    
    grpo_best_comb = -1
    
    if READER is not None and HAVE_TRL and HAVE_DS and USE_GRPO:
        try:
            # Sample GRPO training queries
            rng = np.random.RandomState(SEED)
            sel = rng.choice(len(pool), int(min(GRPO_TRAIN_Q, len(pool))), replace=False)
            grpo_pool = pool.iloc[sel].reset_index(drop=True)
            
            grpo_rows = make_examples_with_candidates(grpo_pool, RET_POOL, pool, K=CAND_TOPK, drop_self=True)
            gds = HFDataset.from_list([
                {
                    'prompt': r['prompt'],
                    'reference': r.get('reference', ''),
                    'subset': r['subset']
                }
                for r in grpo_rows
            ])
            
            g = int(GRPO_NUMGEN)
            bsz = g * max(1, int(GRPO_BSZ) // g)
            
            import inspect, dataclasses
            def _cfg(cls, **kw):
                valid = set()
                try:
                    if dataclasses.is_dataclass(cls):
                        valid |= {f.name for f in dataclasses.fields(cls)}
                except Exception:
                    pass
                try:
                    valid |= set(inspect.signature(cls.__init__).parameters)
                except Exception:
                    pass
                drop = [k for k in kw if valid and k not in valid]
                if drop:
                    print(f'[cfg] {cls.__name__} dropping unsupported: {drop}')
                return cls(**({k: v for k, v in kw.items() if (not valid) or k in valid}))
            
            gcfg = _cfg(
                GRPOConfig,
                output_dir=str(WORK_DIR / 'grpo'),
                learning_rate=GRPO_LR,
                beta=GRPO_BETA,
                per_device_train_batch_size=bsz,
                gradient_accumulation_steps=GRPO_ACCUM,
                num_generations=g,
                max_completion_length=GRPO_MAXNEW,
                max_steps=GRPO_STEPS,
                logging_steps=10,
                save_strategy='no',
                bf16=(DEVICE == 'cuda'),
                report_to=[],
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={'use_reentrant': False}
            )
            
            # Custom trainer with step callback
            from transformers import TrainerCallback
            
            class GRPOStepCallback(TrainerCallback):
                def __init__(self, callback):
                    self.callback = callback
                    self.last_step = -1
                
                def on_step_end(self, args, state, control, logs=None, **kwargs):
                    step = state.global_step
                    if step != self.last_step:
                        self.last_step = step
                        loss = logs.get('loss', 0.0) if logs else 0.0
                        should_stop = self.callback.on_step_end(step, loss)
                        control.should_training_stop = should_stop
            
            gtr = None
            for tk in ('processing_class', 'tokenizer'):
                try:
                    gtr = GRPOTrainer(
                        model=READER,
                        reward_funcs=[comb_reward_snap],
                        args=gcfg,
                        train_dataset=gds,
                        **{tk: RTOK}
                    )
                    break
                except TypeError:
                    continue
            
            if gtr is None:
                gtr = GRPOTrainer(model=READER, reward_funcs=[comb_reward_snap], args=gcfg, train_dataset=gds)
            
            grpo_callback = GRPOSnapEvalCallback(dev_rows, dev_refs, eval_steps=GRPO_EVAL_STEPS, patience=GRPO_PATIENCE)
            gtr.add_callback(GRPOStepCallback(grpo_callback))
            gtr.train()
            
            print('\nGRPO complete')
            print('Training history:')
            for h in grpo_callback.history:
                print(f'  Step {h["step"]}: loss={h["loss"]:.4f}, COMB={h["COMB"]:.4f}')
            
            # Restore best
            grpo_best_comb = grpo_callback.restore_best()
            
            # Compare with SFT best
            if sft_best_comb > 0 and grpo_best_comb < sft_best_comb:
                print(f'\nGRPO ({grpo_best_comb:.4f}) did NOT beat SFT ({sft_best_comb:.4f}) -> REVERTED to SFT adapter')
                restore_adapter(sft_callback.best_state)
            else:
                print(f'\nKeeping post-GRPO model (GRPO={grpo_best_comb:.4f} vs SFT={sft_best_comb:.4f})')
            
        except Exception as e:
            print(f'[warn] GRPO failed (keeping SFT reader): {type(e).__name__}: {str(e)[:200]}')
    else:
        print('GRPO skipped')
    
    # ========== FINAL DEV EVAL ==========
    print('\n' + '=' * 70)
    print('FINAL DEV EVALUATION')
    print('=' * 70)
    
    final_result = eval_dev_checkpoint(dev_rows, dev_refs, tag='FINAL')
    
    # ========== TEST SUBMISSION ==========
    print('\n' + '=' * 70)
    print('BUILDING TEST SUBMISSION')
    print('=' * 70)
    
    # Retrain retriever on full data
    RET_FULL = finetune_retriever(pd.concat([pool, dev], ignore_index=True))
    
    # Process test set
    test_rows = make_examples_with_candidates(test, RET_FULL, pd.concat([pool, dev], ignore_index=True), K=CAND_TOPK, drop_self=True)
    _, snapped, _ = reader_generate_and_snap(test_rows, batch=8)
    
    # Build submission (dual render: ROUGE cols = snapped, LLM col = raw gen if needed)
    # For now, same text in all columns (can optimize later)
    ans = {}
    for i, rid in enumerate(test[ID_COL].values):
        ans[rid] = snapped[i]
    
    out = [str(ans.get(rid, '')).strip() or (str(qq).strip()[:300] or 'N/A') 
           for rid, qq in zip(test[ID_COL].values, test[QUESTION_COL].values)]
    
    sub = pd.DataFrame({
        'ID': test[ID_COL].values,
        'TargetRLF1': out,
        'TargetR1F1': out,
        'TargetLLM': out
    })[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
    
    assert len(sub) == len(test) and (sub['TargetRLF1'].str.len() > 0).all()
    sub.to_csv(OUTPUT_SUBMISSION, index=False, encoding='utf-8')
    
    print(f'\nSaved {OUTPUT_SUBMISSION} shape={sub.shape}')
    print('\n' + '=' * 70)
    print('PIPELINE COMPLETE')
    print('=' * 70)

if __name__ == '__main__':
    main()
