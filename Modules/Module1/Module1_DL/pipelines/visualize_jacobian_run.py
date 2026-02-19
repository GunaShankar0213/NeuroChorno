from Modules.Module1.visualize_jacobian import visualize_jacobian_overlay


def run():

    visualize_jacobian_overlay(
        jac_path=r"Data\outputs\jacobian_map.nii.gz",
        ref_path=r"Data\outputs\dl_registration\fixed_rs.nii.gz",
        out_png=r"Data\outputs\jacobian_overlay_fixed.png",
        slice_axis=2,
        slice_index=None,   # auto mid-slice
        clip_value=0.5,
        alpha=0.55
    )


if __name__ == "__main__":
    run()
