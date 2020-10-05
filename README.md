### User Guide

#### General Description:

The epitope predictor tool takes a file containing one or more amino acid sequences in fasta format
and, using a trained LSTM neural network model, calculates the probability of each amino acid to be
part of an epitope.

All the tool’s output files will be stored inside a directory named `prediction_output` in the input
file’s parent directory. In case a file or directory named `prediction_output` already exist in the input
file’s parent directory, the tool will attempt to create a `prediction_output1` directory, if that also
exists `prediction_output2`, and so on. The user will be alerted of the output directory’s name if it is
not the default.

The tool outputs files in 3 flavors:

1. A text file where line i contains the probability of the i’th amino acid to be in an epitope. The
    file is named after the fasta description of the sequence entry, and has a `.probs` file
    extension. The tool will always output this file for every sequence.
2. A plot of the predicted probability by amino acid position in the sequence. There are several
    additional formatting options for this output discussed below. The plots are stored as `.png`
    picture files named by the fasta description of the sequence. This output format is optional.
3. An annotated replica of the input file, where all amino acids given probability above some
    threshold (user dependent, see below) are capitalized. This file will always be called
    `annotated_output.fasta`, and is an optional output.

**Note:** since the fasta description of sequences decides the filenames of outputs, short descriptions
are preferred. If you are a windows user, avoid unsupported filesystem characters in your
description (see here). For more information about the fasta file format, see here.

Also, since output files are named based on fasta description, a file containing multiple sequences
with the same fasta description or empty description lines could result in unexpected behavior and
faults in the program.

#### Tool's Container:

The tool is encapsulated inside a docker container. As such, it is required to install docker on the
device running the tool. For more information about installing docker, see here.

Next, to load the docker image, either pull it from dockerhub from the repository
`omershapira/epitope-predictor:1.0` or download the image file and extract it with the `docker load`
command.

Finally, the tool is started using the `docker run` command, along with any run options the user
wishes to add. One necessary option is to mount the directory containing the input file as a docker
volume on the container. For more information about docker volumes and their use can be found
here.

#### Tool's Use and Options:

The epitope predictor tool is used much like a UNIX command line function, with the syntax epitope-
predictor [options] [input file path]. This syntax is affixed after the docker run options.


The tool has several options regarding its output:

- The `--visualize` and `--no-visualize` flags toggle the creation of probability to position
    graphs. For further visibility, residues are colored by their probability to be in an epitope
    (higher probability residues get “hotter” colors, and lower probability “colder” colors). Also,
    for every contiguous section of residues above a certain threshold, the position of the first
    residue is marked on the graph itself. The default option is `--no-visualize`.
- The `--annotate` and `--no-annotate` flags toggle the creation of the annotated input file,
    where each amino acid above some threshold is capitalized. The default option is `--no-
    annotate`.
- The user can set the threshold mentioned in the options above using the `--threshold`
    option. The default value is 0.9.
- The `--show-thres` and `--hide-thres` flags toggle drawing the threshold on graphs. This flag
    has no effect in case the `--no-visualize` flag is toggled. The default option is `--hide-thres`.

**Note:** at any time, you can run the tool with the `--help` flag to get a reminder of the tool's optional
parameters and additional info.

#### Run Example:

A typical run of the tool might look like this:

```

docker run --rm --volume /usr/home/research:/data epitope-predictor --visualize --no-annotate --
threshold=0.85 --show-thres /data/sequences.fasta

```

In this example run, the user runs docker from the root user, deleting the container after running it
(`--rm`) and mounting the /usr/home/research directory on the container’s file system as a subfolder
of the root directory named data.

The tool is started with flags instructing it to create graphs of the prediction probability by position
and plot the threshold in those graphs, with the threshold set to 0.85 (see example of graph output
next page). The flags also instruct the tool not to create an annotated fasta file as output. Finally, the
file supplied to the tool is the sequences.fasta file in the container’s /data directory – this means
that the tool actually uses the file /usr/home/research/sequences.fasta in the user’s file system.

Note that the run command is practically the same whether it is called from bash on a Linux
machine, or from Windows Powershell. The only differences are in the path specification between
operating systems and a `sudo` prefix Linux users might need to add to run the container from a root
user.

**Hint:** a common use case is to supply the shell’s current working directory, or one of its sub-
directories to the tool. In such cases, there’s a simple shorthand which supplies the current working
directory: let’s say you want to mount the “seq_data” sub-directory of the working directory, you
can use `--volume $(pwd)/seq_data:/data` on Linux or `--volume ${PWD}\seq_data:/data` on
Windows Powershell.

### Maintainer Guide

**_Software Requirements_**

- Python version 3.x with the following packages installed (listed in the
    requirements.txt):
       ▪ docker==4.3.
       ▪ biopython==1.
       ▪ pytorch==1.6.
- Docker platform installation is required for running cd-hit clustering tool and building
    newer versions of the prediction container.
- If you wish to re-train the model, it is recommended to have a GPU. Also, if you wish
    to re-fetch the data, it is recommended to provide the data fetching script with an
    Eutils API key.

#### Data Fetching Maintenance Guide

**Description**

The data used to train the model was derived from an export of all T-cell essays in the IEDB
(see IEDB export) performed on 22nd June 2020. The IEDB essays contain only the epitope
sequence and an antigen id, and so the antigen sequences themselves were queried from
NCBI using the Eutils web server. The downloaded antigen sequences were stored in fasta
format files, with the epitope sequence of each antigen essay capitalized. Preliminary
duplicate screening and removal was also done during data fetching.

**Design**

The data fetching process consists of two modules:

- _download_data_
    This module contains a single function to download a csv file and save it in the
    provided path.
- _parse_tcell_epitope_


```
The module requires a source path (which should lead to a database export csv file),
and a destination path to store the resulting fasta files in, both provided as
command line arguments.
```
**Execution**

To execute, run the __main__ section in parse_tcell_epitope.py. It is possible to provide a
download URL and directory paths to save both the raw csv and parsed antigen fasta files in.
Before executing this step, it is recommended to configure the “fetch_config” section in the
config.json. This step is also executed as a part of the model training pipeline.

**Configuration**

It is recommended to configure the “fetch_config” section in the config.json file to include
an Entrez Eutils API key and maintainer email. Those values will be sent to NCBI and can
speed up the antigen sequence request rate considerably (see NCBI web service policy).

**Suggestions**

There are two possible methods to increase the size of the models training set for future
runs. As an easy start, the IEDB database is not set in stone, so downloading a more recent
collection of all T-cell essays will surely wield a larger raw basis for training samples.

Moreover, a small portion of the antigen sequences were identified by PIR (protein
information resources) and PRF (protein research foundation) ids, but querying these
databases has proven problematic for an automated script and so they were ignored. Given
an automated method to query sequences from these sites, additional data can be added to
the training set.

### Data Processing

**Description**

Before training the model, the following steps were applied to the downloaded dataset (for
elaboration and explanations, see the algorithm section):


1. Merging identical sequences: The downloaded data consists record sequences with a
    single verified region in each record, meaning that the number of times that a
    sequence is presented in the dataset is equal to the number of verified regions in it.
    We merged the identical records to one representative holding all epitopes on it.
2. Removing highly similar sequences: In order to reduce redundancy, the sequences
    were clustered using an identity threshold of 80% (can be configured), and then only
    the sequence with maximum number of verified regions were chosen out of each
    cluster.
3. Splitting to independent subsets: The dataset was divided to 10 equally sized subsets
    of sequences, while the subsets are comprised of sequences clustered by 50%
    identity threshold. The number of subsets can be controlled and could also be used
    for cross validation.

**Design**

The data preparation code is comprised of 4 modules:

- _Model_
    Data abstraction of:
    - _Epitope_ : Amino acid sequence with verified regions capitalized.
    - _EpitopesDataset_ : A dataset of Epitope objects.
    - _EpitopesClusters_ : Clusters of Epitope objects.
- _Preprocess_
    set of utilities for data preprocessing:
    - _clustering_ :
       ▪ cluster_records_by_identity - util for clustering EpitopesDataset by
          identity threshold.
    - _datasets_preprocessing_ :
       ▪ get_epitopes_with_max_verified_regions - processing
          EpitopesClusters to get a list of Epitope objects with maximum
          number verified regions from each cluster.


```
▪ split_epitopes_clusters_to_cv_datasets - splitting EpitopesClusters to
EpitopesDatasets as cross-validation subsets and saving each dataset
as FASTA file.
```
- _Preprocessor_ – An object that performs all the preparation steps mentioned
    above (more details in the diagram below).
- _run_processing_
The main script, which gets the user arguments, loads the configurations, and
operate the Preprocessor.
- _Tests_
unit-tests for the model and preprocess components.

Below is a diagram of the main flow of run_proccess:


The numbers on the direction arrows that are coming out of the ‘Init and execute Preprocessor’ box,
indicate the order of the steps in the directed boxes.


**Execution**

These instructions are for running the preprocessing step as a standalone and assumes the
previous step (data fetching) has already been performed. Also, this step executes as part of
the model training pipeline (model_main).

- Clone the repository with: git clone https://github.com/idanbudin93/epitope-
    predictor.git
- Make sure that the Docker program is running in the background and configured to a
    Docker-hub user (or another container image library that contains the cd-hit image).
- Execute: run the following command –
    o python < _pathToRepository>_ /epitope-predictopr/run_processing --input_files
       <inputFile1.fasta> <inputFile2.fasta>... <inputFileX.fasta> --config
       < _pathToConfigFile_ >--use_rand_seed
          ▪ The input file argument value is a list of whitespace-separated input
             files paths in FASTA format.
          ▪ The --config argument is optional, the default configuration file is
             loaded from the root directory of the repository.
          ▪ The --use_rand_seed flag is optional and is used for setting constant
             random seed (which results in deterministic test-train subsets
             splitting).
- Output – The output is the independent datasets saved in the ‘output’ folder in the
    root directory of the repository.
       o Output files are FASTA formatted.
       o Each output file name is ending with its group number (before the file
          extension).

**Configuration**

It’s possible to modify the configuration file or provide a new one, as long as the keys and
values types are not changed.

The following settings can be found in the configuration file (config.json) under
_Processing_config_ :


- homologs_threshold – The Identity threshold for considering sequences as homologs
    (a float between 0 and 1).
- homologs_clustering_word_size – The word size parameter for cd-hit (see cd-hit
    user’s guide for choosing it).
- cv_groups_clustering_threshold – The Identity threshold for clustering epitopes
    together for the cross-validation datasets splitting (float between 0 and 1).
- cv_fold – The number of cross-validation datasets to split the data into (int between
    0 and the number of clusters in the dataset).
- temp_output_dir – Path to the temporary output files directory (str representing a
    valid absolute or relative path).
- cd_hit_docker_name - The cd-hit Docker image name to pull (str representing an
    image name, that must be available for pulling from the logged in Docker user on the
    machine).
- random_seed – A random seed to use when –use_random_seed flag is given (an int /
    str / bytes / bytesarray).
- output_template – A template for the paths to the output files with a placeholder
    for the cross-validation group number (str representing a valid absolute or relative
    path).

### Model and Training

**Description**

This module contains the model used for the project – an LSTM network with configurable
dimensions and number of layers, and various utilities for preparing the data and training
the model.

**Design**

The model and training step is comprised of 3 modules:

- **_lstm_model_**
    in this module you can find the _LSTMtagger_ class and its forward implementation,
    based on the architecture that is discussed in detail in the Algorithm section. You can


```
control its parameters in the config.json file. Also in this module, are methods that
are relevant for the embedding part, data labeling, and for getting the probabilities
for each tagging option.
```
- **_train_and_test_**
    holds 2 classes:
    1. _Class Trainer:_ an abstract class, abstracting the tasks of training a model. It
       provides methods at multiple levels:
          o fit – training and testing for multiple epochs.
          o train_epoch/test_epoch – training and testing for single epoch.
          o train_batch/test_batch - training and testing for single batch.
    2. _Class LSTM_trainer_ : an implementation of the abstract class Trainer. Holds
       special accuracy calculation _calc_accuracy_ , and _avg_loss_fn_ , a function that
       returns a function that calculates average of a given loss (more on the specific
       calculations on the algorithm’s part).
- **model_main**
    This module contains the whole pipeline of building our trained model, starting from
    downloading the samples and processing them through to the training itself. During
    the model main run, the maintainer is notified of steps completed in the pipeline
    and intermediate results. One can control training parameters in the config file,
    which now contains our choice of default parameters.

**Execute**

To run it, open a cmd on windows computer, navigate to the folder where this file in, and
call “python model_main.py”. No command line arguments or extra parameters are
necessary, as all parameters are taken from the config.json file. For quicker training, it is
recommended to make sure a GPU with CUDA is available for pytorch to use.

**Config Parameters**

The “lstm_model_config” section in the config.json file contains the, and it holds the values
we used for our final model. Though, they can’t be controlled there. Here are some
explanations for each of them:


- Checkpoint_file – specifies a path and name for the file where training checkpoints
    will be saved to.
- training_plot_name – specifies a name for the plot the training produces.
- Hidden_dim - controls the number of cells in each lstm layer.
- n_layers – controls the number of layers.
- bidirectional – controls whether LSTM will be bidirectional (1) or not (0).
- dropout – controls the dropout probability.
- lr – controls the learning rate.
- num_epochs – controls the number of epochs.
- early_stopping – the training will stop after this number of epochs with no
    improvement.
- seq_len – the size of samples.
- batch_size – controls the size of batches of samples.
- train_test_ratio – specifies the portion of the data to be taken to train.

**Suggestions:**

- There was an attempt to integrate word embedding to the model, which yielded
    unsatisfactory results. You can still try it out, and you can consider also trying more
    types of embedding, such as the N-gram model for different N parameters.
- We implemented batching but, in the end, didn’t use it. Consider trying it too.

### User Interface

**Design:**

The current user interface of the tool is a simple wrapper of the trained model,
encapsulated in a Docker container. It is based on the python:3-slim image for reduce
container size, and all python dependencies are specified in a requirements text loaded to
the container.

The file listing of the user interface is as follows:


- _interface_main:_ main entry point of the container, performs most file IO operations
    and handles user options.
- _interface_plot:_ utility file for graphing visualized output and storing it to file.
- _lstm_model:_ a copy of lstm_model from the train and test module is imported to the
    container to perform the actual prediction.
- _Misc. files:_ the interface relies a Dockerfile and a python requirements file for
    building the container, as well as on a saved_model.pt file, containing the trained
    model, for predictions.

The tool’s semblance of a shell command is provided by the click library, and all added
parameters to the tool should be added as click command options. Make sure to add a
“help” parameter to every option to display the correct information when calling `run
docker epitope-predictor --help`.

Any additional code file added should be stored in the `interface_code` directory, and after
any change in the tool, the Docker image should be rebuilt.

A word about GUI: while creating a Graphic User Interface was one of the suggested goals of
the project, it was ultimately dropped for a simple reason: the tool is shipped inside a
Docker container, and running is done via a shell program. As such, it would be far simpler
to add a few additional parameters and run the tool in one line. While some functional
changes might warrant a more interactive approach like a GUI, the expansions suggested
here do not.

**Execution:**

The execution guide here refers to building a Docker image of the tool, and generally follows
the instructions provided here.

All the required files to build the docker image are provided in the interface directory, and
an image can be built in a computer running docker by the command:

`docker build -t <tag> <interface path>`

With <tag> replaced by an image name of the builder’s choice, and <interface path>
replaced by an absolute or relative path to the interface directory. The command runs the
same on Windows and Linux operating systems, only Linux users might need to run the


command from root user (with `sudo`), and specify network use for building the image with
the option `--network host`.

**Configuration:**

There are no inherent configuration options to building the interface container except for
those provided by Docker. For options to users when running the interface, refer to the user
guide.

**Suggestions:**

While the epitope prediction tool delivers on its basic premise of predicting the probabilities
of an amino acid (aa) to be a part of an epitope given an aa sequence, there are several
ways in which the tool can be expanded. The possibilities described here can fall roughly
into 2 categories: input options and output options.

There are, of course, other changes which can be considered, such as allowing the user to
integrate additional experimental data to the prediction process, such as indicating
sequence positions with lower, higher probability to be in an epitope or providing epitopes
found in homolog sequences. However, such options require exposing the user to the in-
depth prediction process of the network, whether possible, and are out of the scope of this
section.

- Expanding input options:
    Currently, the tool takes a path to a fasta file and runs the prediction routine on each
    sequence in the file. A natural extension to that will be accepting other formats, such
    as text files containing a single sequence or XML files to allow a more seamless
    integration of the tool with the output of the NCBI’s Eutils.
    Another natural extension is to allow setting a directory as input, such that all
    relevant files in that directory will be passed through the tool.
- Expanding output options:
    While the tool has no shortage of output options, two helpful additions would be to
    let the user choose the location of the output directory, instead of using the input
    file’s parent directory as default. However, note that since the tool is in a container,


```
the output parent directory should be in a volume, whether it is the volume
containing the input file or a different one.
Moreover, it might be preferable to let the user decide on some pattern (regex,
grep, glob, etc.) to extract an output file name from fasta description lines. Normally,
NCBI extracted fasta files contain a long description line segmented into different
parts by pipe (|) characters, and biopython’s method to segment the description has
been found lacking. If a better format is presented to the users and enforced by
them, it could remove a lot of hassle from using the tool.
```
- Docker suggestions: Since the prediction functionality relies on some third-party
    packages, the size of the container has a lower bound of at least 200 Megabytes at
    the time of writing. Due to size considerations of the tool container, the pytorch
    version uploaded to the container does not support CUDA GPU acceleration.
    In cases where GPU acceleration matters more than file size, the container can be
    rebuilt with CUDA (note that currently it is not possible to import pytorch without
    CUDA from a requirements file, so the requirements file contains a commented line
    to import pytorch with CUDA, while the Dockerfile explicitly imports pytorch with no
    CUDA).
    Alternatively, a smaller docker image can be achieved by changing the image to be
    based on Linux alpine (python:3-alpine image), however this requires adjusting
    several python packages.


