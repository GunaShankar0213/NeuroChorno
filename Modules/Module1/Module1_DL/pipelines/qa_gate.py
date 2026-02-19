from Modules.Module1.qa_deformation import qa_check

def run_qa():

    accept, metrics = qa_check(
        flow_path=r"Data\outputs\dl_registration\flow_dl.npy",
        fixed_path=r"Data\outputs\dl_registration\fixed_rs.nii.gz",
        warped_path=r"Data\outputs\dl_registration\warped_dl.nii.gz",
        out_json=r"Data\outputs\qa_metrics.json"
    )

    print("QA ACCEPTED:", accept)
    print(metrics)

    if not accept:
        print("⚠️ Trigger ANTs SyN fallback")

if __name__ == "__main__":
    run_qa()
