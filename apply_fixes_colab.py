"""
Colab'da çalıştırılacak yama scripti.
Bu dosyayı Colab notebook'ta bir hücreye yapıştırıp çalıştırın:

    !python apply_fixes_colab.py

Veya hücre başına %%writefile ile kaydedin, sonra çalıştırın.
"""
import re
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def patch_file(filepath, patches):
    """Apply find-replace patches to a file."""
    abs_path = os.path.join(PROJECT_DIR, filepath)
    if not os.path.exists(abs_path):
        print(f"❌ File not found: {abs_path}")
        return False
    
    with open(abs_path, 'r') as f:
        content = f.read()
    
    original = content
    for i, (find, replace, description) in enumerate(patches):
        if find in content:
            content = content.replace(find, replace, 1)
            print(f"  ✅ Patch {i+1}: {description}")
        elif replace in content:
            print(f"  ⏭️  Patch {i+1}: Already applied — {description}")
        else:
            print(f"  ❌ Patch {i+1}: Target text not found — {description}")
            print(f"     Looking for: {repr(find[:80])}...")
            return False
    
    if content != original:
        with open(abs_path, 'w') as f:
            f.write(content)
        print(f"  💾 Saved: {filepath}")
    else:
        print(f"  ℹ️  No changes needed: {filepath}")
    return True

print("=" * 60)
print("  APPLYING FIXES TO test_cvusa.py, utils.py, dataset_rgbd.py")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# FIX 1: test_cvusa.py — CLI --use_rgbd override protection
# ═══════════════════════════════════════════════════════════════
print("\n📁 Patching test_cvusa.py...")

test_patches = [
    # Patch 1a: Save CLI value before config override
    (
        "opt = parser.parse_args()\n###load config###",
        "opt = parser.parse_args()\n_cli_use_rgbd = opt.use_rgbd  # Save CLI value before config can override it\n###load config###",
        "Save CLI --use_rgbd before config override"
    ),
    # Patch 1b: Use CLI value with OR logic
    (
        "# Load use_rgbd from config - this is critical for model/data compatibility\nopt.use_rgbd = config.get('use_rgbd', False)",
        "# Load use_rgbd from config — but CLI --use_rgbd flag takes priority\nopt.use_rgbd = _cli_use_rgbd or config.get('use_rgbd', False)",
        "CLI --use_rgbd takes priority over config"
    ),
]
patch_file("test_cvusa.py", test_patches)

# ═══════════════════════════════════════════════════════════════
# FIX 2: utils.py — Don't overwrite use_rgbd if already True
# ═══════════════════════════════════════════════════════════════
print("\n📁 Patching utils.py...")

utils_patches = [
    (
        "    # RGBD support\n    opt.use_rgbd = config.get('use_rgbd', False)",
        "    # RGBD support — preserve CLI flag if already True\n    if not getattr(opt, 'use_rgbd', False):\n        opt.use_rgbd = config.get('use_rgbd', False)",
        "Preserve CLI use_rgbd in load_network()"
    ),
]
patch_file("utils.py", utils_patches)

# ═══════════════════════════════════════════════════════════════
# FIX 3: dataset_rgbd.py — Index-based labels for CVUSA
# ═══════════════════════════════════════════════════════════════
print("\n📁 Patching dataset_rgbd.py...")

dataset_patches = [
    # Patch 3a: CVUSADataset labels
    (
        """                self.images.append(filepath)
                # Extract numeric ID from filename for consistent labeling
                # e.g., "12345.jpg" -> 12345
                basename = os.path.splitext(filename)[0]
                # Handle filenames like "12345_sat.jpg" or just "12345.jpg"
                numeric_part = basename.split('_')[0]
                if numeric_part.isdigit():
                    self.labels.append(int(numeric_part))
                else:
                    # Fallback: use hash for consistency
                    self.labels.append(hash(basename) % 10000000)""",
        """                self.images.append(filepath)
                # Use sequential index as label — in CVUSA the i-th query
                # image is paired with the i-th gallery image (sorted order)
                self.labels.append(len(self.images) - 1)""",
        "CVUSADataset: index-based labels"
    ),
    # Patch 3b: CVUSARGBDDataset labels
    (
        """                    self.rgb_images.append(rgb_path)
                    self.depth_images.append(depth_path)
                    # Extract numeric ID from filename for consistent labeling
                    numeric_part = basename.split('_')[0]
                    if numeric_part.isdigit():
                        self.labels.append(int(numeric_part))
                    else:
                        self.labels.append(hash(basename) % 10000000)""",
        """                    self.rgb_images.append(rgb_path)
                    self.depth_images.append(depth_path)
                    # Use sequential index as label — in CVUSA the i-th query
                    # image is paired with the i-th gallery image (sorted order)
                    self.labels.append(len(self.rgb_images) - 1)""",
        "CVUSARGBDDataset: index-based labels"
    ),
]
patch_file("dataset_rgbd.py", dataset_patches)

# ═══════════════════════════════════════════════════════════════
# FIX 4: test_cvusa.py — Vectorized metric computation
# ═══════════════════════════════════════════════════════════════
print("\n📁 Patching test_cvusa.py (vectorized metrics)...")

# Check if the old per-query matmul pattern exists
test_path = os.path.join(PROJECT_DIR, "test_cvusa.py")
with open(test_path, 'r') as f:
    test_content = f.read()

OLD_METRIC_MARKER = "score     = (gf @ qf.unsqueeze(1)).squeeze(1).numpy()"
NEW_METRIC_MARKER = "scores = torch.mm(q_feat, g_feat.t()).numpy()"

if OLD_METRIC_MARKER in test_content:
    # Find and replace the entire old metric section
    old_section = '''    def _eval_query(qf, ql, gf, gl):
        """Pure-numpy metric computation (CPU). Returns (ap, first_hit_rank_1indexed, cmc_array)."""
        score     = (gf @ qf.unsqueeze(1)).squeeze(1).numpy()  # CPU torch.mm
        index     = np.argsort(score)[::-1]
        good_idx  = np.argwhere(gl == ql).flatten()
        junk_idx  = np.argwhere(gl == -1).flatten()
        # remove junk entries
        mask      = np.in1d(index, junk_idx, invert=True)
        index     = index[mask]
        ngood     = len(good_idx)
        if ngood == 0:
            return 0.0, -1, None
        mask2     = np.in1d(index, good_idx)
        rows_good = np.argwhere(mask2).flatten()
        if len(rows_good) == 0:
            return 0.0, -1, None
        n_clean   = len(index)
        cmc       = np.zeros(n_clean, dtype=np.float32)
        cmc[rows_good[0]:] = 1.0
        first_rank = int(rows_good[0]) + 1  # 1-indexed
        ap = 0.0
        for i in range(ngood):
            d_recall  = 1.0 / ngood
            precision = (i + 1) / (rows_good[i] + 1)
            old_prec  = i / rows_good[i] if rows_good[i] != 0 else 1.0
            ap       += d_recall * (old_prec + precision) / 2
        return ap, first_rank, cmc

    CMC_accum = np.zeros(n_gallery, dtype=np.float64)
    ap_list   = []
    rank_list = []
    valid_q   = 0

    for i in range(len(q_lbl)):
        ap_tmp, rank_tmp, cmc_tmp = _eval_query(q_feat[i], q_lbl[i], g_feat, g_lbl)
        if cmc_tmp is None: continue

        if (i + 1) % 100 == 0 or i == 0:
            print(f'  Query {i+1}/{len(q_lbl)} evaluated...', flush=True)
            sys.stdout.flush()

        valid_q += 1
        ap_list.append(ap_tmp * 100)
        rank_list.append(rank_tmp)
        n_c = len(cmc_tmp)
        if n_c >= n_gallery:
            CMC_accum += cmc_tmp[:n_gallery]
        else:
            CMC_accum[:n_c] += cmc_tmp
            CMC_accum[n_c:] += cmc_tmp[-1]

    print(f'\\n  Done. Valid: {valid_q}/{len(q_lbl)}', flush=True)'''

    new_section = '''    # ── Step 1: Compute full similarity matrix in one shot ────────────
    print('  Computing similarity matrix...', flush=True)
    sys.stdout.flush()
    scores = torch.mm(q_feat, g_feat.t()).numpy()
    print(f'  Similarity matrix computed: {scores.shape}', flush=True)

    # ── Step 2: Sort once (descending) ────────────────────────────────
    print('  Sorting similarities...', flush=True)
    sys.stdout.flush()
    sorted_indices = np.argsort(-scores, axis=1)
    print('  Sorting done.', flush=True)

    # ── Step 3: Evaluate per query (fast — no more matmul inside loop) ─
    CMC_accum = np.zeros(n_gallery, dtype=np.float64)
    ap_list   = []
    rank_list = []
    valid_q   = 0
    interrupted = False
    junk_mask_gl = (g_lbl == -1)
    eval_start = time.time()

    try:
        for i in range(n_query):
            ql = q_lbl[i]
            index = sorted_indices[i]

            if junk_mask_gl.any():
                keep = ~np.in1d(index, np.where(junk_mask_gl)[0])
                index = index[keep]

            good_idx = np.where(g_lbl == ql)[0]
            ngood = len(good_idx)
            if ngood == 0:
                continue

            match_mask = np.in1d(index, good_idx)
            rows_good  = np.where(match_mask)[0]
            if len(rows_good) == 0:
                continue

            n_clean = len(index)
            cmc = np.zeros(n_clean, dtype=np.float32)
            cmc[rows_good[0]:] = 1.0
            first_rank = int(rows_good[0]) + 1

            ap = 0.0
            for j in range(ngood):
                d_recall  = 1.0 / ngood
                precision = (j + 1) / (rows_good[j] + 1)
                old_prec  = j / rows_good[j] if rows_good[j] != 0 else 1.0
                ap       += d_recall * (old_prec + precision) / 2

            valid_q += 1
            ap_list.append(ap * 100)
            rank_list.append(first_rank)
            if n_clean >= n_gallery:
                CMC_accum += cmc[:n_gallery]
            else:
                CMC_accum[:n_clean] += cmc
                CMC_accum[n_clean:] += cmc[-1]

            if (i + 1) % 500 == 0 or i == 0 or (i + 1) == n_query:
                elapsed = time.time() - eval_start
                eta = elapsed / (i + 1) * (n_query - i - 1) if i > 0 else 0
                print(f'  Query {i+1}/{n_query}  valid={valid_q}  '
                      f'elapsed={elapsed:.0f}s  ETA={eta:.0f}s', flush=True)
                sys.stdout.flush()
                if use_wandb and valid_q > 0:
                    partial_cmc = CMC_accum / valid_q
                    wandb.log({
                        'eval/progress_pct': round((i + 1) / n_query * 100, 1),
                        'eval/valid_queries': valid_q,
                        'eval/partial_Recall@1': float(partial_cmc[0]) * 100,
                        'eval/partial_mAP': float(np.mean(ap_list)),
                    })

    except KeyboardInterrupt:
        print(f'\\n⚠️  Interrupted at query {i+1}/{n_query}. Reporting partial results...', flush=True)
        interrupted = True

    print(f'\\n  Done. Valid: {valid_q}/{len(q_lbl)}', flush=True)'''

    # Also fix the n_gallery line to add n_query
    test_content = test_content.replace(
        "    n_gallery = len(g_lbl)\n    print(f'  Query: {len(q_lbl)}  Gallery: {n_gallery}  Feat-dim: {q_feat.shape[1]}', flush=True)",
        "    n_query   = len(q_lbl)\n    n_gallery = len(g_lbl)\n    print(f'  Query: {n_query}  Gallery: {n_gallery}  Feat-dim: {q_feat.shape[1]}', flush=True)"
    )

    if old_section in test_content:
        test_content = test_content.replace(old_section, new_section)
        with open(test_path, 'w') as f:
            f.write(test_content)
        print("  ✅ Vectorized metric computation applied")
    else:
        print("  ❌ Could not find old metric section to replace")
elif NEW_METRIC_MARKER in test_content:
    print("  ⏭️  Vectorized metrics already applied")
else:
    print("  ❌ Neither old nor new metric pattern found")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ALL PATCHES APPLIED!")
print("=" * 60)
print("\n🚀 Now run your test with --use_wandb flag:")
print("""
!python test_cvusa.py \\
  --name lpn_square_test \\
  --test_dir /content/cvpr2017_cvusa/test \\
  --depth_dir /content/cvpr2017_cvusa_depth/test \\
  --query_folder query_satellite \\
  --gallery_folder gallery_drone \\
  --use_rgbd \\
  --which_epoch last \\
  --gpu_ids 0 \\
  --use_wandb
""")
