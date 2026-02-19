from Modules.Module1.flow_to_nifti import flow_to_nifti

def run():

    flow_to_nifti(
        flow_path=r"Data\outputs\dl_registration\flow_dl.npy",
        ref_img_path=r"Data\outputs\dl_registration\fixed_rs.nii.gz",
        out_path=r"Data\outputs\dl_registration\warp_field_dl.nii.gz" ## final out 
    )

if __name__ == "__main__":
    run()
