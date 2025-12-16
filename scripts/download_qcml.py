# from google.cloud import storage
import os
from re import I
import tensorflow as tf
import tensorflow_datasets as tfds

# def download_public_file(bucket_name, source_blob_name, destination_file_name):
#     """Downloads a public blob from the bucket."""
#     # bucket_name = "your-bucket-name"
#     # source_blob_name = "storage-object-name"
#     # destination_file_name = "local/path/to/file"

#     storage_client = storage.Client.create_anonymous_client()

#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#     blob.download_to_filename(destination_file_name)

#     print(
#         "Downloaded public blob {} from bucket {} to {}.".format(
#             source_blob_name, bucket.name, destination_file_name
#         )
#     )

# https://zenodo.org/records/14859804

if __name__ == "__main__":
    """
    Download:
    dft_atomic_numbers: 11 (3 GB)
    dft_positions: 11 (7 GB)
    """

    LOCAL_DATA_DIR = "/tmp"
    QCML_DATA_DIR = "gs://qcml-datasets/tfds"
    GCP_PROJECT = "deepmind-opensource"

    # Installation of the used 'gcloud': https://cloud.google.com/sdk/docs/install

    # ===========================
    # No authentication necessary
    # ===========================
    # Alternatively, see https://cloud.google.com/docs/authentication/gcloud.
    os.system("gcloud config set auth/disable_credentials True")

    # =============================================
    # Download only needed builder configs
    # =============================================
    # Directory structure:
    # gs://qcml-datasets/tfds/qcml/dft_atomic_numbers/1.0.0/...
    # gs://qcml-datasets/tfds/qcml/dft_positions/1.0.0/...

    # dft_atomic_numbers
    if not os.path.exists(f"{LOCAL_DATA_DIR}/qcml/dft_atomic_numbers/"):
        os.system(f"mkdir -p {LOCAL_DATA_DIR}/qcml/dft_atomic_numbers/")
        os.system(
            f"gcloud storage cp -r "
            f"gs://qcml-datasets/tfds/qcml/dft_atomic_numbers/1.0.0 "
            f"{LOCAL_DATA_DIR}/qcml/dft_atomic_numbers/ --project={GCP_PROJECT}"
        )
    else:
        print(f"{LOCAL_DATA_DIR}/qcml/dft_atomic_numbers/ already exists")

    # dft_positions
    if not os.path.exists(f"{LOCAL_DATA_DIR}/qcml/dft_positions/"):
        os.system(f"mkdir -p {LOCAL_DATA_DIR}/qcml/dft_positions/")
        os.system(
            f"gcloud storage cp -r "
            f"gs://qcml-datasets/tfds/qcml/dft_positions/1.0.0 "
            f"{LOCAL_DATA_DIR}/qcml/dft_positions/ --project={GCP_PROJECT}"
        )
    else:
        print(f"{LOCAL_DATA_DIR}/qcml/dft_positions/ already exists")

    # =============================================
    # Load only the requested datasets and zip them
    # =============================================
    read_config = tfds.ReadConfig(interleave_cycle_length=1)

    dft_atomic_numbers_ds = tfds.load(
        "qcml/dft_atomic_numbers",
        split="full",
        data_dir=LOCAL_DATA_DIR,
        read_config=read_config,
    )
    dft_positions_ds = tfds.load(
        "qcml/dft_positions",
        split="full",
        data_dir=LOCAL_DATA_DIR,
        read_config=read_config,
    )

    ds = tf.data.Dataset.zip((dft_atomic_numbers_ds, dft_positions_ds))
    example_numbers, example_positions = next(iter(ds))

    print("Example:")

    print("dft_atomic_numbers:", example_numbers)
    print("dft_positions:", example_positions)

    # # =============================================
    # # Example 1: Feature group 'dft_force_field'
    # # =============================================
    # # TFDS directory structure: <data_dir>/<dataset>/<builder_config>/<version>.
    # os.system(f'mkdir -p {LOCAL_DATA_DIR}/qcml/dft_force_field/')
    # os.system(
    #     f'gcloud storage cp -r {QCML_DATA_DIR}/qcml/dft_force_field/1.0.0'
    #     f' {LOCAL_DATA_DIR}/qcml/dft_force_field/ --project={GCP_PROJECT}'
    # )
    # force_field_ds = tfds.load(
    #     'qcml/dft_force_field', split='full', data_dir=LOCAL_DATA_DIR
    # )
    # force_field_iter = iter(force_field_ds)
    # example = next(force_field_iter)
    # print(example)

    # # ===================================================================
    # # Example 2: Combine 'dft_force_field' with 'dft_d4_correction'
    # # ===================================================================
    # os.system(f'mkdir -p {LOCAL_DATA_DIR}/qcml/dft_d4_correction/')
    # os.system(
    #     f'gcloud storage cp -r {QCML_DATA_DIR}/qcml/dft_d4_correction/1.0.0'
    #     f' {LOCAL_DATA_DIR}/qcml/dft_d4_correction/ --project={GCP_PROJECT}'
    # )
    # # Note the read config to keep the same record order in both datasets.
    # read_config = tfds.ReadConfig(interleave_cycle_length=1)
    # force_field_ds_for_zip = tfds.load(
    #     'qcml/dft_force_field',
    #     split='full',
    #     data_dir=LOCAL_DATA_DIR,
    #     read_config=read_config,
    # )
    # d4_correction_ds_for_zip = tfds.load(
    #     'qcml/dft_d4_correction',
    #     split='full',
    #     data_dir=LOCAL_DATA_DIR,
    #     read_config=read_config,
    # )
    # zipped_ds = tf.data.Dataset.zip(
    #     force_field_ds_for_zip, d4_correction_ds_for_zip
    # )
    # zipped_iter = iter(zipped_ds)

    # # The example contains one tuple element (feature dict) per input dataset.
    # example = next(zipped_iter)
    # print('atomic_numbers from first', example[0]['atomic_numbers'])
    # print('d4_energy from second', example[1]['d4_energy'])
    # # The feature 'key_hash' can be used to verify the correct example order.
    # print('Matching key_hashes', [t['key_hash'] for t in example])
