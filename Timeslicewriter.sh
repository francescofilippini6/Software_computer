#!/bin/zsh
 #
 #
 # \author mdejong
 #
 version=1.0
 script=${0##*/}
 
 # ------------------------------------------------------------------------------------------
 #
 #                         Utility script to test JRadomTimesliceWriter.
 #
 # ------------------------------------------------------------------------------------------
 
 if [ -z $JPP_DIR ]; then
     echo "Variable JPP_DIR undefined."
     exit
 fi
 
 source $JPP_DIR/setenv.sh $JPP_DIR
 
 set_variable     DEBUG           2
 set_variable     DIR             $JPP_DIR/examples/JTimeslice
 set_variable     WORKDIR         /sps/km3net/users/ffilippi/ML/random_noise
 
 if ( do_usage $* ); then
     usage "$script [working directory]"
 fi
 
 case $# in
     1) WORKDIR=$1;;
 esac
 
 set_variable  BACKGROUND_HZ              10e3 
 set_variable  numberOfSlices             1000
 set_variable  RECYCLING                  5 100e3
 
 if ( ! reuse_file $WORKDIR/timeslice.root); then
 
     $DIR/JRandomTimesliceWriter \
	 -B $BACKGROUND_HZ \
	 -o $WORKDIR/timeslice.root \
	 -n $numberOfSlices \
	 -N "$RECYCLING" \
	 -d $DEBUG --!
 fi
 
 for (( N = 1; $N <= $RECYCLING[1]; ++N )); do
 
     JPlot1D \
	 -w 1200x600 \
	 -f "$WORKDIR/timeslice.root:h0\[0\]" \
	 -f "$WORKDIR/timeslice.root:h0\[${N}\]" \
	 -\> "time [ns]" \
	 -\^ "number of hits [a.u.]" \
	 -T "$N"
 done
