# Workflow

From `/pscratch/sd/y/yhzhang/VAECloud`, on an interactive node. Run each script with `bash` (not `sbatch`). Output goes to the terminal.

`GAUSSIAN.sh` writes `LIST_GAUSS.ascii`, `MATRIX_GAUSS.ascii`, `CORRELATION_GAUSS.pdf`, and `VALUE_GAUSS.ini`. VALUE uses `--covariance=full`, which reads `MATRIX.ascii`. Use `--covariance=gauss` only if that year's CHECK does not report full `cholesky_success`.

## CELL/COVARIANCE

`CONFIG.sh` is the prep step. It writes the OneCovariance inputs (`LENS.ascii`, `SOURCE.ascii`, `GALAXY.ascii`, `Cell_*.ascii`, ...) and both ini files (`GAUSSIAN.ini`, `CONFIG.ini`). `GAUSSIAN.sh` and `MATRIX.sh` only run OneCovariance. They are independent after prep and can run in either order.

Do **not** run `MATRIX.sh` before `CONFIG.sh` (or `REORDER.sh` on this overwrite). `MATRIX.sh` no longer builds inputs.

This overwrite already has Cell tables. Use `REORDER.sh` instead of `CONFIG.sh` so those tables are not regenerated.

```text
INFO, LENS/SOURCE VAE+PCA, CELL SHIFT/SCALE/CORRECT/VAE
        |
        +-- TRUNCATE ---------------- CELL TT/TE/EE PCA
        |
        +-- CONFIG  (from scratch)
        |     or REORDER  (this overwrite; keep existing Cell_*.ascii)
        |
        +-- GAUSSIAN.sh  and  MATRIX.sh  (either order, or both)
        |        |
        +--------+-- CHECK
                     |
                     +-- VALUE  (after PCA and Gaussian CHECK)
```

## 1. INFO (already done)

```bash
bash INFO/COSMOLOGY.sh
bash INFO/SURVEY.sh
bash INFO/DENSITY.sh
bash INFO/GALAXY.sh
bash INFO/ALIGNMENT.sh
bash INFO/MAGNIFICATION.sh
```

## 2. LENS / SOURCE VAE and 150-PC PCA (already done)

```bash
bash LENS/Y1/VAE.sh
bash LENS/Y10/VAE.sh
bash SOURCE/Y1/VAE.sh
bash SOURCE/Y10/VAE.sh

bash LENS/Y1/PCA.sh
bash LENS/Y10/PCA.sh
bash SOURCE/Y1/PCA.sh
bash SOURCE/Y10/PCA.sh
```

## 3. CELL SHIFT / SCALE / CORRECT / VAE (already done)

```bash
bash CELL/Y1/TT/SHIFT.sh
bash CELL/Y1/TE/SHIFT.sh
bash CELL/Y1/EE/SHIFT.sh
bash CELL/Y10/TT/SHIFT.sh
bash CELL/Y10/TE/SHIFT.sh
bash CELL/Y10/EE/SHIFT.sh

bash CELL/Y1/TT/SCALE.sh
bash CELL/Y1/TE/SCALE.sh
bash CELL/Y1/EE/SCALE.sh
bash CELL/Y10/TT/SCALE.sh
bash CELL/Y10/TE/SCALE.sh
bash CELL/Y10/EE/SCALE.sh

bash CELL/Y1/TT/CORRECT.sh
bash CELL/Y1/TE/CORRECT.sh
bash CELL/Y1/EE/CORRECT.sh
bash CELL/Y10/TT/CORRECT.sh
bash CELL/Y10/TE/CORRECT.sh
bash CELL/Y10/EE/CORRECT.sh

bash CELL/Y1/TT/VAE.sh
bash CELL/Y1/TE/VAE.sh
bash CELL/Y1/EE/VAE.sh
bash CELL/Y10/TT/VAE.sh
bash CELL/Y10/TE/VAE.sh
bash CELL/Y10/EE/VAE.sh
```

## 4. Truncate n(z) (run)

Independent of covariance prep.

```bash
bash LENS/Y1/TRUNCATE.sh
bash SOURCE/Y1/TRUNCATE.sh
bash LENS/Y10/TRUNCATE.sh
bash SOURCE/Y10/TRUNCATE.sh
```

## 5. Covariance prep

**This overwrite:** reorder the existing Cell tables. Do not run `CONFIG.sh`.

```bash
bash CELL/Y1/COVARIANCE/REORDER.sh
bash CELL/Y10/COVARIANCE/REORDER.sh
```

**From scratch:** `CONFIG.sh` rebuilds n(z), bias tables, theory `Cell_*.ascii` (already lexsorted), `GAUSSIAN.ini`, and `CONFIG.ini`. Then `REORDER.sh` is unnecessary.

```bash
bash CELL/Y1/COVARIANCE/CONFIG.sh
bash CELL/Y10/COVARIANCE/CONFIG.sh
```

## 6. Covariance (after prep)

Gaussian (`*_GAUSS.*`) and full (`LIST.ascii`, `MATRIX.ascii`) write different files and can run in parallel.

```bash
bash CELL/Y1/COVARIANCE/GAUSSIAN.sh
bash CELL/Y10/COVARIANCE/GAUSSIAN.sh
bash CELL/Y1/COVARIANCE/MATRIX.sh
bash CELL/Y10/COVARIANCE/MATRIX.sh
```

## 7. CHECK (after both matrices for that year)

```bash
bash CELL/Y1/COVARIANCE/CHECK.sh
bash CELL/Y10/COVARIANCE/CHECK.sh
```

## 8. CELL PCA (after TRUNCATE)

```bash
bash CELL/Y1/TT/PCA.sh
bash CELL/Y1/TE/PCA.sh
bash CELL/Y1/EE/PCA.sh
bash CELL/Y10/TT/PCA.sh
bash CELL/Y10/TE/PCA.sh
bash CELL/Y10/EE/PCA.sh
```

## 9. VALUE (after PCA and full CHECK)

```bash
bash VALUE/Y1/SINGLE/VALUE.sh
bash VALUE/Y1/DOUBLE/VALUE.sh
bash VALUE/Y1/TRIPLE/VALUE.sh
bash VALUE/Y10/SINGLE/VALUE.sh
bash VALUE/Y10/DOUBLE/VALUE.sh
bash VALUE/Y10/TRIPLE/VALUE.sh
```

## 10. Optional latent traversal

```bash
bash PLOT/Y1/VARIABLE.sh
bash PLOT/Y10/VARIABLE.sh
```
