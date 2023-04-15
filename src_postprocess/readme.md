Output visualization and postprocessing
--------------------------------
**To visualize the raw stain concentration maps:**  
 
* Use method `visualize_raw_stain_conc_map()` in `src_postprocess/vis_different_thresholds.py`.  
* Set the following variables:  
	* `text_2_rgb_id`: a dictionary with key = biomarker name, and value = [< alternative RGB color to use in biomarker visualization>, < channel indx>, < one-hot encoding of channel index>, < stain reference RGB color>].  
	* `root`: path of root directory containing all evaluation results.  
	* `conc_dir_name`: folder inside <`root`> directory. All model predictions are in `<root>/<conc_dir_name>`.  
	* `im_dir`: path of folder containing the test images.  
* The method output will be in `<root>/<conc_dir_name>_vis`.
* See example in the source file.

**To visualize the thresholded stain concentration maps at different thresholds:**  

* Use method `visualize_raw_stain_conc_map_by_threshold()` in `src_postprocess/vis_different_thresholds.py`.  
* Set the following variables:  
	* `text_2_rgb_id`: a dictionary with key = biomarker name, and value = [< alternative RGB color to use in biomarker visualization>, < channel indx>, < one-hot encoding of channel index>, < stain reference RGB color>].  
	* `root`: path of root directory containing all evaluation results.  
	* `conc_dir_name`: folder inside `<root>` directory. All model predictions are in `<root>/<conc_dir_name>`.  
	* `im_dir`: path of folder containing the test images.  
	* `threshold_arr`: array of threshold values to use.  
* The method output will be in `<root>/<conc_dir_name>_thresholds_vis`.  
* See example in the source file.
* **Note:**  
The visualizations are used to select thresholds based on validation set.  
The hysteresis thresholding, has 2 thresholds: high-thresh. and low-thresh. Select high-thresh to reflect confident regions that belong to the stain, and select low-thresh to reflect how much the regions of confidence are expanded to cover the stained area.

**To get the stain segmentation maps:**  

* Use method `argmax_all_conc_thresh_mask_hysteresis()` in `src_postprocess/seg_argmax.py`
* Set the following variables:  
	* `text_2_rgb_id`: a dictionary with key = biomarker name, and value = [< alternative RGB color to use in biomarker visualization>, < channel indx>, < one-hot encoding of channel index>, < stain reference RGB color>].  
	* `stain_names`: the list of biomarker names excluding the background.  
	* `size_thresh_dict`: a dictionary with  key = biomarker name, and value = size in pixels. predicted components with smaller size are discarded. This helps to get rid of prediction noise corresponding to noise in tissue, in the form of tiny scattered predictions.  
	* `root`: path of root directory containing all evaluation results.  
	* `conc_dir_name`: folder inside `<root>` directory. All model predictions are in `<root>/<conc_dir_name>`.
	* `im_dir`: path of folder containing the test images
	* `thresh_dict_high` and `thresh_dict_low`: dictionaries of high and low thresholds needed for hysteresis thresholding.  
* The method output will be in `<root>/<conc_dir_name>_seg`.  
* See example in the source file.
* Note: since the annotation uses dots at the approximate center of cells which can often stain free, especially in the case of tumor cells. To be able to make fair evaluation, in the postprocessing we apply a small dilation and fill the holes.

**To visualize the stain segmentation maps:**

* Use the methods `visualize_mask_img()` and `visualize_all_stains_in_single_mask()` in `src_postprocess/vis_seg.py`.  
* The method output will be in `<root>/<conc_dir_name>_seg_vis`.  
* See example in the source file.
 