from os import path

from docker import DockerClient


def cluster_records_by_identity(
        docker_client: DockerClient,
        cd_hit_img_id: str,
        fasta_input_path: str,
        output_path_without_clstr_ext: str,
        identity_threshold: float,
        word_size: int
) -> str:
    input_dir = path.dirname(fasta_input_path)
    output_dir = path.dirname(output_path_without_clstr_ext)

    input_fname = path.basename(fasta_input_path)
    output_fname = path.basename(output_path_without_clstr_ext)

    cmd = f'cd-hit -i /input/{input_fname} -o /output/{output_fname}' \
          + f' -c {identity_threshold} -n {word_size} -G 0 -aL 1.0'

    stdout_and_err_bytes = \
        docker_client.containers.run(
            image=cd_hit_img_id,
            volumes=[
                f'{input_dir}:/input',
                f'{output_dir}:/output'
            ],
            remove=True,
            detach=False,
            stdout=True,
            stderr=True,
            command=cmd
        )

    return stdout_and_err_bytes.decode('utf-8')
