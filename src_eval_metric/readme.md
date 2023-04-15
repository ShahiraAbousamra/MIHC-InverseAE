Output evaluation
--------------------------------
**To evaluate the F-score:**

* Compare to a ground truth multi-class dot maps.  
* Use example in `src_eval_metric/eval_fscore_96patches.py`.  

**To evaluate the image reconstruction:** 

* Use the code in `src_eval_metric/96patches_eval_reconstruction.py`.  
* The code evaluates the Structural Similarity (SSIM), Peak Signal-to-Noise Ratio (PSNR), Mean Squared Error (mse), Learned Perceptual Image Patch Similarity (LPIPS).
 