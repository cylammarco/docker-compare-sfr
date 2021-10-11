FROM ubuntu:20.04
LABEL maintainer="MCL <lam@tau.ac.il>"

# Workaround https://unix.stackexchange.com/questions/2544/how-to-work-around-release-file-expired-problem-on-a-local-mirror
RUN echo "Acquire::Check-Valid-Until \"false\";\nAcquire::Check-Date \"false\";" | cat > /etc/apt/apt.conf.d/10no--check-valid-until

# Basic software installation
RUN apt-get update
RUN apt-get install -y wget nano python3.8 python3-pip git gfortran
RUN pip3 install scipy~=1.7 numpy~=1.21 matplotlib~=3.4 dynesty~=1.1 emcee~=3.1 astro-sedpy~=0.2 corner~=2.2
RUN pip3 install git+https://github.com/dfm/python-fsps.git
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

# Set environment variable
ENV SPS_HOME="/home/sfr/git/fsps"
RUN git clone --depth 1 https://github.com/cconroy20/fsps.git $SPS_HOME

# create "home directory"
WORKDIR /home/sfr
RUN mkdir pipe3d
RUN mkdir prospector
RUN mkdir ppxf
RUN mkdir example

COPY run_all_examples.sh .
