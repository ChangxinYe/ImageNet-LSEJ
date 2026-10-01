# Source-image selection

These scripts record how the 12,000 benchmark images were chosen from the ILSVRC2012 training set. They require access to the original ImageNet images under the applicable [ImageNet terms](https://www.image-net.org/download.php). The released `splits/*.csv` files are the authoritative list of selected images and split assignments; ordinary dataset use does not require running this pipeline.

The pipeline runs in this order:

1. `s2_imagenet_size_stats.py` counts candidate images by short-side resolution. The 1,000 px threshold yielded 13,091 candidates from 969 classes.
2. `s3_copy_ge_1000px_images.py` copies candidates with a short side of at least 1,000 px, preserving class directories.
3. `s4_center_crop_1000px.py` center-crops candidates to 1000x1000 PNG without resizing.
4. `s5_evaluate_puzzle_suitability.py` scores each cropped image and selects the top 12,000. The published selection used the defaults `--top-k 12000 --low-std 10.0 --low-gradient 3.0`.

The suitability score is a weighted sum of low-information-piece score (30%), largest connected low-information component score (20%), piece-uniqueness score (20%), mean-gradient score (15%), and piece-detail score (15%). Exact feature definitions and tie ordering are in `s5_evaluate_puzzle_suitability.py`. The published candidate ranking and measurements are in [s5_scores.csv](s5_scores.csv), with aggregate parameters in [s5_summary.json](s5_summary.json).

Each script accepts explicit source and output directories via `--help`. The stage scripts write their intermediate images and reports to the chosen output directories; keep these outputs outside the Git repository. The final accepted image filenames were checked against `splits/all.csv`: both contain the same 12,000 images.
