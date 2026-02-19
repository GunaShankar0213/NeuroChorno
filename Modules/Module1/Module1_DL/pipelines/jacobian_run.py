from Modules.Module1.jacobian import compute_jacobian_determinant


def run():

    compute_jacobian_determinant(
        warp_path=r"Data\outputs\dl_registration\warp_field_dl.nii.gz",
        out_path=r"Data\outputs\jacobian_map.nii.gz",
        log_jacobian=True
    )


if __name__ == "__main__":
    run()
