#!/bin/zsh
#
# \author mdejong
#
version=1.0
script=${0##*/}

# ------------------------------------------------------------------------------------------
#
#                         Utility script to test JTriggerProcessor.
#
# ------------------------------------------------------------------------------------------


if [ -z $JPP_DIR ]; then
    echo "Variable JPP_DIR undefined."
    exit
fi

source $JPP_DIR/setenv.sh $JPP_DIR

set_variable  DEBUG          2
set_variable  WORKDIR        /sps/km3net/users/ffilippi/ML/random_noise
set_variable  DETECTOR       /sps/km3net/users/ffilippi/ML/caliMupage.detx
#set_variable  DETECTOR       $JPP_DATA/km3net_reference.detx 115 strings configuration
set_variable  TRIGGER        $JPP_DATA/trigger_parameters_arca.txt
set_variable  BACKGROUND_HZ  10e3 600 60 7 0.8 0.08
set_variable  RECYCLING      10 100e3
set_variable  numberOfSlices 10000

if ( do_usage $* ); then
    usage "$script [detector file [trigger file [working directory]]]"
fi

case $# in
    3) set_variable WORKDIR  $3;&
    2) set_variable TRIGGER  $2;&
    1) set_variable DETECTOR $1;;
esac

set_variable  INPUT_FILE     $WORKDIR/timeslice_10k_1.root
set_variable  OUTPUT_FILE    $WORKDIR/trigger_processor_10k_1.root

if [[ ! -f $DETECTOR ]]; then
    JDetector.sh $DETECTOR
fi

if ( ! reuse_file $INPUT_FILE ); then

    echo "Generating random background."

    print_variable     DETECTOR INPUT_FILE BACKGROUND_HZ
    check_input_file   $DETECTOR
    check_output_file  $INPUT_FILE

    timer_start

    JRandomTimesliceWriter \
	-a $DETECTOR                  \
	-o $INPUT_FILE                \
	-n $numberOfSlices            \
	-B "$BACKGROUND_HZ"           \
	-N "$RECYCLING"               \
	-d $DEBUG                          

    timer_stop
    timer_print

fi

if ( ! reuse_file $OUTPUT_FILE ); then

    echo "Processing data."

    print_variable     DETECTOR TRIGGER INPUT_FILE OUTPUT_FILE
    check_input_file   $DETECTOR $TRIGGER $INPUT_FILE
    check_output_file  $OUTPUT_FILE

    timer_start

    JTriggerProcessor \
	-a $DETECTOR                  \
	-f $INPUT_FILE                \
	-o $OUTPUT_FILE               \
	-@ $TRIGGER                   \
	-C JDAQTimesliceL0            \
	-d $DEBUG
  
    timer_stop
    timer_print

fi

JPrintTree -f $OUTPUT_FILE
