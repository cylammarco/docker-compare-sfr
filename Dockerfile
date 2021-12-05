FROM ubuntu:20.04
LABEL maintainer="MCL <lam@tau.ac.il>"

# Workaround https://unix.stackexchange.com/questions/2544/how-to-work-around-release-file-expired-problem-on-a-local-mirror
RUN echo "Acquire::Check-Valid-Until \"false\";\nAcquire::Check-Date \"false\";" | cat > /etc/apt/apt.conf.d/10no--check-valid-until

# Sort out time zone and location
# https://askubuntu.com/questions/909277/avoiding-user-interaction-with-tzdata-when-installing-certbot-in-a-docker-contai
ENV TZ=Asia/Jerusalem
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Basic software installation
RUN apt-get update
RUN apt-get install -y wget nano python3.8 python3-pip git gfortran

# Set environment variable
ENV SPS_HOME="/home/sfr/git/fsps"
RUN git clone --depth 1 https://github.com/cconroy20/fsps.git $SPS_HOME

# Install python software
RUN pip3 install scipy~=1.7 numpy~=1.21 matplotlib~=3.4 dynesty~=1.1 emcee~=3.1 astro-sedpy~=0.2 corner~=2.2 astropy speclite pyparsing~=2.4

# Install fsps with Padova model
FFLAGS="-DMIST=0 -DPADOVA=1" python -m pip install fsps --no-binary fsps

# Make sure python refers to python3.8
RUN ln -s $(which python3.8) /usr/bin/python

# Compile TLUSTY and SYNSPEC
WORKDIR /home/sfr/git

# Install Pipe3D
RUN git clone https://gitlab.com/pipe3d/pyPipe3D.git
RUN pip3 install pyPipe3D/.

# Install Prospector (stable version)
RUN git clone https://github.com/bd-j/prospector.git
RUN pip3 install prospector/.

# Install ppxf
# https://www-astro.physics.ox.ac.uk/~mxc/software/#ppxf
RUN pip3 install ppxf
RUN pip3 install plotbin

# create "home directory"
WORKDIR /home/sfr
RUN mkdir pipe3d
RUN mkdir prospector
RUN mkdir ppxf
RUN mkdir example
RUN mkdir shared_folder
RUN mkdir synthetic_spectra

COPY run_all_examples.sh .
