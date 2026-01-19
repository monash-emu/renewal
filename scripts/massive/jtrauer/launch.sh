#! /usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

sbatch --export=SCRIPT_DIR=$SCRIPT_DIR $SCRIPT_DIR/run_countries_active.sh
