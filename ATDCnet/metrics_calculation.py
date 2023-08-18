import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio,mean_squared_error


def calculate_metrics_ssim_psnr_all(generated_image_path, ground_truth_image_path, resize_size=(256, 256)):
    generated_image_list = os.listdir(generated_image_path)
    error_list_ssim, error_list_psnr,error_list_ssim_YCBCR, error_list_psnr_YCBCR,error_list_ssim_GRAY, error_list_psnr_GRAY,error_list_mse = [], [],[], [],[], [],[]

    for img in generated_image_list:
        label_img = img
        generated_image = os.path.join(generated_image_path, img)
        ground_truth_image = os.path.join(ground_truth_image_path, label_img)
        #print(generated_image)
        generated_image = cv2.imread(generated_image)
        generated_image = cv2.resize(generated_image, resize_size)
        generated_image_YCRCB=cv2.cvtColor(cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2YCR_CB)
        generated_image_GRAY=cv2.cvtColor(generated_image,cv2.COLOR_BGR2GRAY)

        ground_truth_image = cv2.imread(ground_truth_image)
        ground_truth_image = cv2.resize(ground_truth_image, resize_size)
        ground_truth_image_YCBCR= cv2.cvtColor(cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2YCR_CB)
        ground_truth_image_GRAY = cv2.cvtColor(ground_truth_image,cv2.COLOR_BGR2GRAY)


        # calculate SSIM
        error_ssim, diff_ssim = structural_similarity(generated_image, ground_truth_image, full=True, multichannel=True)
        error_ssim_YCRCB, diff_ssim_YCRCB = structural_similarity(generated_image_YCRCB, ground_truth_image_YCBCR, full=True, multichannel=True)
        error_ssim_GRAY, diff_ssim_GRAY = structural_similarity(generated_image_GRAY, ground_truth_image_GRAY, full=True, multichannel=True)
        error_list_ssim.append(error_ssim)
        error_list_ssim_YCBCR.append(error_ssim_YCRCB)
        error_list_ssim_GRAY.append(error_ssim_GRAY)

        # generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
        # ground_truth_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)

        # calculate PSNR
        error_psnr = peak_signal_noise_ratio(generated_image, ground_truth_image)
        error_psnr_YCBCR = peak_signal_noise_ratio(generated_image_YCRCB, ground_truth_image_YCBCR)
        error_psnr_GRAY = peak_signal_noise_ratio(generated_image_GRAY, ground_truth_image_GRAY)
        error_list_psnr.append(error_psnr)
        error_list_psnr_YCBCR.append(error_psnr_YCBCR)
        error_list_psnr_GRAY.append(error_psnr_GRAY)


        # calculate MSE
        error_mse=mean_squared_error(generated_image,ground_truth_image)
        error_list_mse.append(error_mse)




    return np.mean(np.array(error_list_ssim)), np.mean(np.array(error_list_psnr)),np.mean(np.array(error_list_ssim_YCBCR)),np.mean( np.array(error_list_psnr_YCBCR)),np.mean(np.array(error_list_ssim_GRAY)), \
           np.mean(np.array(error_list_psnr_GRAY)),np.mean(np.array(error_list_mse))





