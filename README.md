suggestions:
for 4x4 pixel sized AI generated images:
python pixelart_grid_fit_downsample.py .\victory_arcDeTriumph_2.png .\outputs\test-arc.png  --min-cell 4 --max-cell 4  --print-grid --workers 0 --jitter 1 --size-tweak 1 --addDateTag --majority-min-frac 0.65 --thr 12 --qbits 4 

prompt to generage images in Gemini:
"a 17th century vibrant Colonial city in Americas. There is a statue of Cristopher Columb in the town square. Merchants bringing goods at docks. There is a fort wall with soldiers watching over. pixelart style with vibrant colors befitting Imperial Ambitions style. use 8 pixels per pixelart cell grid."


usage: pixelart_grid_fit_downsample.py [-h] [--min-cell MIN_CELL] [--max-cell MAX_CELL] [--cell-step CELL_STEP] [--offset-step OFFSET_STEP] [--sample-step SAMPLE_STEP] [--thr THR] [--qbits QBITS]
                                       [--majority-min-frac MAJORITY_MIN_FRAC] [--alpha-cutoff ALPHA_CUTOFF] [--jitter JITTER] [--size-tweak SIZE_TWEAK] [--alpha ALPHA] [--workers WORKERS] [--print-grid]
                                       [--anchor ANCHOR] [--addDateTag]
                                       input_png output_png

Fit pixel-art grid and downsample (multiprocess).

positional arguments:
  input_png
  output_png

options:
  -h, --help            show this help message and exit
  --min-cell MIN_CELL   Minimum grid cell size (pixels) to consider during grid fitting. Set this close to the expected pixel size in the source image to speed up search.
  --max-cell MAX_CELL   Maximum grid cell size (pixels) to consider during grid fitting. For faster, more reliable fits, keep the [min,max] range tight (e.g., 10..14).
  --cell-step CELL_STEP
                        Step size when scanning candidate cell sizes. Use 1 for accuracy; use 2–4 for speed if you are unsure and searching broadly.
  --offset-step OFFSET_STEP
                        Step size for scanning x/y offsets within a cell (0..cell-1). 1 is most accurate; 2–4 is faster. If you already know the grid is aligned, larger steps are usually fine.
  --sample-step SAMPLE_STEP
                        Stride for sampling pixels inside each cell *during grid scoring only*. 1 = use all pixels (best accuracy). 2–3 can speed up grid fitting on large images with minimal quality loss.        
  --thr THR             RGB distance threshold (0..~441) used to keep pixels as inliers around the dominant color cluster within each cell. Lower = sharper edges but may drop thin/diagonal strokes; higher =      
                        more tolerant to noise/anti-aliasing but can smear. Typical: 12–22. Start at 16–18 for clean pixel art.
  --qbits QBITS         Quantization bits per color channel used to estimate the dominant color cluster (mode bin). Higher = more bins (more sensitive, can split similar colors); lower = coarser bins (more       
                        stable, but may merge shades). Typical: 4–6. Default 5 is a good balance.
  --majority-min-frac MAJORITY_MIN_FRAC
                        Minimum fraction of pixels that must be retained as inliers for a cell. If the threshold-based inliers fall below this, the algorithm falls back to safer rules (dominant bin or a robust   
                        MAD filter). Lowering this (e.g., 0.35–0.45) helps preserve thin/diagonal contours; raising it makes fills more stable.
  --alpha-cutoff ALPHA_CUTOFF
                        Ignore pixels with alpha < cutoff (0..255) when sampling cells. Use 0 to include fully transparent pixels; keep default for sprites with transparent backgrounds.
  --jitter JITTER       Per-cell sampling window jitter radius in pixels during refinement. The algorithm tries offsets in [-jitter,+jitter] for both x and y to better align with the true grid. 0 disables        
                        refinement. Typical: 1. Values >=2 can improve difficult cases but increase runtime and can cause drifting unless --anchor is used.
  --size-tweak SIZE_TWEAK
                        Allow per-cell sampling window size adjustment by +/- this many pixels during refinement. Helps when the source grid is not perfectly uniform (capture/scale artifacts). 0 disables.        
                        Typical: 0–1. Values >=2 increase runtime and risk contour loss unless anchored.
  --alpha ALPHA         Dispersion penalty weight during refinement. Higher penalizes mixed-color windows, reducing smearing and discouraging the optimizer from sampling across edges. Typical: 0.20–0.60. If      
                        edges smear, increase; if thin strokes vanish, consider lowering slightly or reducing jitter/size-tweak and using --anchor.
  --workers WORKERS     Number of worker processes (0=auto, 1=disable parallelism).
  --print-grid          Print the best detected grid parameters (cell size and offsets) and the fit score.
  --anchor ANCHOR       Anchor penalty weight during refinement. Penalizes candidates that deviate from the base (non-jittered) cell color. This prevents the per-cell optimizer from drifting into flatter
                        regions and accidentally erasing thin/diagonal contours. 0 disables. Typical: 0.4–1.0 when using jitter>=1 or size-tweak>=1.
  --addDateTag
