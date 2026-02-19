from Modules.Module1.DL_register import run_dl_registration


def run_dl_stage():

    fixed_path = r"Data\outputs\bias_corrected\T0\result_image_n4.nii.gz"
    moving_path = r"Data\outputs\affine\T1_affine_aligned.nii.gz"
    model_path = r"model_store\TransMorph-diff\transmorph_large\TransMorphLarge.pth.tar"
    out_dir = r"Data\outputs\dl_registration"

    warped_path, flow_path = run_dl_registration(
        fixed_path=fixed_path,
        moving_path=moving_path,
        model_path=model_path,
        out_dir=out_dir
    )

    print("DL Registration Done")
    print("Warped:", warped_path)
    print("Flow:", flow_path)


if __name__ == "__main__":
    run_dl_stage()
