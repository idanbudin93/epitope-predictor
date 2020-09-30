# **User Guide**

**General description:**

The epitope predictor tool takes a file containing one or more amino acid sequences in fasta format and, using a trained LSTM neural network model, calculates the probability of each amino acid to be part of an epitope.

All the tool&#39;s output files will be stored inside a directory named `prediction_output` in the input file&#39;s parent directory. In case a file or directory named `prediction_output` already exist in the input file&#39;s parent directory, the tool will attempt to create a `prediction_output1` directory, if that also exists `prediction_output2`, and so on. The user will be alerted of the output directory&#39;s name if it is not the default.

The tool outputs files in 3 flavors:

1. A text file where each line I contains the probability of the i&#39;th amino acid to be in an epitope. The file is named after the fasta description of the sequence entry, and has a `.probs` file extension. The tool will always output this file for every sequence.
2. A plot of the predicted probability by amino acid position in the sequence. There some several additional formatting options for this output discussed below. The plots are stored as `.png` picture files named by the fasta description of the sequence. This output format is optional.
3. An annotated replica of the input file, where all amino acids given probability above some threshold (user dependent, see below) are capitalized. This file will always be called `annotated_output.fasta`, and is an optional output.

**Note:** since the fasta description of sequences decides the filenames of outputs, short descriptions are preferred. If you are a windows user, avoid unsupported filesystem characters in your description (see [here](https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file)). For more information about the fasta file format, see [here](http://prodata.swmed.edu/promals/info/fasta_format_file_example.htm).

Also, since output files are named based on fasta description, a file containing multiple sequences with the same fasta description or empty description lines could result in unexpected behavior and faults in the program.

**Tool container:**

The tool is encapsulated inside a docker container. As such, it is required to install docker on the device running the tool. For more information about installing docker, see [here](https://docs.docker.com/get-docker/).

Next, to load the docker image, either pull it from dockerhub by the tag name [XXX] or download the image file and extract it with the `docker load` command.

Finally, the tool is started using the `docker run` command, along with any run options the user wishes to add. One necessary option is to mount the directory containing the input file as a docker volume on the container. For more information about docker volumes and their use can be found [here](https://docs.docker.com/storage/volumes/).

**Tool use and options:**

The epitope predictor tool is used much like a UNIX command line function, with the syntax [predictor] [options] [input file path]. This syntax is affixed after the docker run options of course.

The tool has several options regarding its output:

- The `--visualize` and `--no-visualize` flags toggle the creation of probability to position graphs. For further visibility, residues are colored by their probability to be in an epitope (higher probability residues get &quot;hotter&quot; colors, and lower probability &quot;colder&quot; colors). Also, for every contiguous section of residues above a certain threshold, the position of the first residue is marked on the graph itself. The default option is `--no-visualize`.
- The `--annotate` and `--no-annotate` flags toggle the creation of the annotated input file, where each amino acid above some threshold is capitalized. The default option is `--no-annotate`.
- The user can set the threshold mentioned in the options above using the `--threshold` option. The default value is 0.9.
- The `--show-thres` and `--hide-thres` flags toggle drawing the threshold on graphs. This flag has no effect in case the `--no-visualize` flag is toggled. The default option is `--hide-thres`.

**Note:** at any time, you can run the tool with the `--help` flag to get a reminder of the tools optional parameters and additional info.

**Run example:**

A typical run of the tool might look like this:

```

docker run --rm --volume /usr/home/research:/data [predictor] --visualize --no-annotate --threshold=0.925 --show-thres /data/sequences.fasta

```

In this example run, the user runs docker from the root user, deleting the container after running it (`--rm`) and mounting the /usr/home/research directory on the container&#39;s file system as a subfolder of the root directory named data.

The container run by docker is [predictor], and it is run with flags instructing it to create graphs of the prediction probability by position and plot the threshold in those graphs, with the threshold set to 0.925. The flags also instruct the tool not to create an annotated fasta file as output. Finally, the file supplied to the tool is the sequences.fasta file in the container&#39;s /data directory – this means that the tool actually uses the file /usr/home/research/sequences.fasta in the user&#39;s file system.

Note that the run command is practically the same whether it is called from bash on a Linux machine, or from Windows Powershell. The only differences are in the path specification for between operating systems and a `sudo` prefix Linux users might need to add to run the container from a root user.

**Hint:** a common use case is to supply the shell&#39;s current working directory, or one of its sub-directories to the tool. In such cases, there&#39;s a simple shorthand which supplies the current working directory: let&#39;s say you want to mount the &quot;seq\_data&quot; sub-directory of the working directory, you can use `--volume $(pwd)/seq_data:/data` on Linux or `--volume ${PWD}\seq_data:/data` on Windows.

# **Maintainer&#39;s Guide**

**Data Fetching**

**Description**

The data used to train the model was derived from an export of all T-cell essays in the IEDB (see [IEDB export](https://www.iedb.org/database_export_v3.php)) performed on 22nd June 2020. The IEDB essays contain only the epitope sequence and an antigen id, and so the antigen sequences themselves were queried from NCBI using the Eutils web server. The downloaded antigen sequences were stored in fasta format files, with the epitope sequence of each antigen essay capitalized. Preliminary duplicate screening and removal was also done during data fetching.

**Design**

The data fetching process consists of a single module, which requires a source path (which should lead to a database export csv file), and a destination path to store the resulting fasta files in, both provided as command line arguments.

The only required package to download the data is biopython (version == 1.78) but providing an NCBI Eutils API key and maintainer email are recommended to speed up the antigen sequence request rate (see [NCBI web service policy](https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/)).

**Suggestions**

There are two possible methods to increase the size of the models training set for future runs. As an easy start, the IEDB database is not set in stone, so downloading a more recent collection of all T-cell essays will surely wield a larger raw basis for training samples.

Moreover, a small portion of the antigen sequences were identified by PIR ([protein information resources](https://proteininformationresource.org/)) and PRF ([protein research foundation](https://www.prf.or.jp/index-e.html)) ids, but querying these databases has proven problematic for an automated script and so they were ignored. Given an automated method to query sequences from these sites, additional data can be added to the training set.

**Data Preparation**

**Description**

Before training the model, the following steps were applied to the downloaded dataset:

1. **Merging Identical Sequences –** The downloaded data consists record sequences with a single verified region in each record, meaning that the number of times that a sequence is presented in the dataset is equal to the number of verified region in it. Merging the records while keeping the verified regions not only shortens the training time by reducing the dataset, but much more importantly, solve a critical issue with not doing so, which will best be explained by an example: Given a sequence with X different verified regions, each verified region will be labeled as such only in one record, and as a result the model will receive incorrect feedback for the prediction X-1 times.
2. **Removing Highly Similar Sequences –** In order to reduce redundancy, the sequences were clustered using an identity threshold of 80% (can be configured), and then only the sequence with maximum number of verified regions was chosen out of each cluster. This reduces the overrepresented proteins in the dataset, which might bias the training of the model.
3. **Splitting to Cross Validation Subsets –** The dataset was divided to 10 equally sized subsets of sequences, while the subsets are comprised of sequences clustered by 50% identity threshold. This ensures that the test subsets will never contain a sequence with 50% or above that was used for training, and by that makes the training and test sets independent as possible from each other.

**Design**

The data preparation code is comprised of 4 modules:

- **Model** – data abstraction of:
  - `Epitope`: aa sequence with verified regions capitalized.
  - `EpitopesDataset`: a dataset of Epitope objects.
  - `EpitopesClusters`: Clusters of Epitope objects
- **Preprocess** – set of utils for preprocessing the data:
  - `clustering`:
    - `cluster_records_by_identity` - util for clustering `EpitopesDataset` by identity threshold.
  - `datasets_preprocessing`:
    - `get_epitopes_with_max_verified_regions` - processing `EpitopesCluster`s to get a list of `Epitope` objects with maximum number verified regions from each cluster.
    - `split_epitopes_clusters_to_cv_datasets` - splitting `EpitopesClusters` to `EpitopesDataset`s as cross-validation subsets and saving each dataset as FASTA file.
  - Preprocessor – An object that performs all the preparation steps mentioned above (more details in the diagram below).
- **run\_processing** – The main script, which gets the user arguments, loads the configurations, and operate the Preprocessor.
- **Tests** – unit-tests for the model and preprocess components.

**Instructions**

- **Requirements -**
  - Python version 3.x with the following packages installed (listed in the requirements.txt):
    - `docker==4.3.1`
    - `biopython==1.78`
  - Docker platform installation is required for running cd-hit clustering tool.
- **Execution -**
  - Clone the repository with: `git clone https://github.com/idanbudin93/epitope-predictor.git`
  - Make sure that the Docker program is running in the background and configured to a Docker-hub user (or another container image library that contains the cd-hit image).
  - Execute: run the following command –
    ```C
    python <pathToRepository>/epitope-predictopr/run_processing --input_files <nputFile1.fasta> <inputFile2.fasta>… <inputFileX.fasta> --config <pathToConfigFile> --use_rand_seed
    ```
      - The `--input_files` argument value is a list of whitespace-separated input files paths in FASTA format.
      - The `--config` argument is optional, the default configuration file is loaded from the root directory of the repository.
      - The `--use_randseed` flag is optional and is used for setting constant random seed (which results in deterministic test-train subsets splitting).
  - Output – The output is the cross-validation datasets saved in the `output` folder in the root directory of the repository.
    - Output files are FASTA formatted.
    - Each output file name is ending with its group number (before the file extension).
- **Configurations -**
  - It&#39;s possible to modify the configuration file or provide a new one, as long as the keys are not changed nor the values types.
  - The following settings can be found in the configuration file:
    - `homologs_threshold` – The Identity threshold for considering sequences as homologs (a `float` between `0` and `1`).
    - `homologs_clustering_word_size` – The word size parameter for cd-hit (see [cd-hit user&#39;s guide](http://www.bioinformatics.org/cd-hit/cd-hit-user-guide.pdf) for choosing it).
    - cv_groups_clustering_threshold – The Identity threshold for clustering epitopes together for the cross-validation datasets splitting (`float` between `0` and `1`).
    - `cv_fold` – The number of cross-validation datasets to split the data into (int between `0` and the number of clusters in the dataset).
    - `temp_output_dir` – Path to the temporary output files directory (str representing a valid absolute or relative path).
    - `cd_hit_docker_name` - The cd-hit docker image name to pull (str representing an image name, that must be available for pulling from the logged in Docker user on the machine).
    - `random_seed` – A random seed to use when `–use\_random\_seed` flag is given (an `int` / `str` / `bytes` / `bytesarray`)
    - `output_template` – A template for the paths to the output files with a placeholder for the cross-validation group number (str representing a valid absolute or relative path with a placeholder).

**Model**

**User Interface**

While the epitope prediction tool delivers on its basic premise of predicting the probabilities of an amino acid (aa) to be a part of an epitope given an aa sequence, there are several ways in which the tool can be expanded. The possibilities described here can fall roughly into 2 categories: input options and output options.

There are, of course, other changes which can be considered, such as allowing the user to integrate additional experimental data to the prediction process, such as indicating sequence positions with lower, higher probability to be in an epitope or providing epitopes found in homolog sequences. However, such options require exposing the user to the in-depth prediction process of the network, whether possible, and are out of the scope of this section.

**Expanding input options:**

Currently, the tool takes a path to a fasta file and runs the prediction routine on each sequence in the file. A natural extension to that will be accepting other formats, such as text files containing a single sequence or XML files to allow a more seamless integration of the tool with the output of the NCBI&#39;s Eutils.

Another natural extension is to allow setting a directory as input, such that all relevant files in that directory will be passed through the tool.

**Expanding output options:**

While the tool has no shortage of output options, two helpful additions would be to let the user choose the location of the output directory, instead of using the input file&#39;s parent directory as default. However, note that since the tool is in a container, the output parent directory should be in a volume, whether it is the volume containing the input file or a different one.

Moreover, it might be preferable to let the user decide on some pattern (regex, grep, glob, etc.) to extract an output file name from fasta description lines. Normally, NCBI extracted fasta files contain a long description line segmented into different parts by pipe (|) characters, and biopython&#39;s method to segment the description has been found lacking. If a better format is presented to the users and enforced by them, it could remove a lot of hassle from using the tool.

**Technical design choices:**

The current user interface of the tool is a simple wrapper of the trained model, encapsulated in a docker container. The docker image is based on the [python:3-slim](https://hub.docker.com/_/python) image for reduce container size, and all python dependencies are specified in a requirements text loaded to the docker.

Since the prediction functionality relies on some third-party packages, the size of the container has a lower bound of at least 200 Megabytes at the time of writing. Due to size considerations of the tool container, the pytorch version uploaded to the docker does not support CUDA GPU acceleration. In cases where GPU acceleration matters more than file size, the docker can be rebuilt with CUDA (note that currently it is not possible to import pytorch without CUDA from a requirements file, so the requirements file contains a commented line to import pytorch with CUDA, while the Dockerfile explicitly imports pytorch with no CUDA).

Alternatively, a smaller docker image can be achieved by changing the image to be based on Linux alpine (python:3-alpine image), however this requires adjusting several python packages.

The tool&#39;s semblance of a shell command is provided by the [click](https://click.palletsprojects.com/en/7.x/) library, and all added parameters to the tool should be added as click command options. Make sure to add a &quot;help&quot; parameter to every option to display the correct information when calling `run docker <predictor> --help`.

Any additional code file added should be stored in the `interface_code` directory, and after any change in the tool, the docker image should be rebuilt.

A word about GUI: while creating a Graphic User Interface was one of the suggested goals of the project, it was ultimately dropped for a simple reason: the tool is shipped inside a docker container, and running is done via a shell program. As such, it would be far simpler to add a few additional parameters and run the tool in one line. While some functional changes might warrant a more interactive approach like a GUI, the expansions suggested here do not.
