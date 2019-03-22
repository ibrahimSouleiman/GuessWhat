#! /bin/zsh

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 environment_name [OPTIONS] command [arg1..argn]" >&2
    echo "OPTIONS:" >&2
    echo "  -h, --help        Show this message" >&2
    echo "  -p, --cuda-path   Specify the path where cuda is installed (default: /usr/local/cuda-8.0)" >&2
    exit 1
fi

cuda_path="/usr/local/cuda-8.0"

while [[ $1 == -* ]]; do
    case "$1" in
        -p|--cuda-path)
            cuda_path="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 environment_name [OPTIONS] command [arg1..argn]"
            echo "OPTIONS:"
            echo "  -h, --help        Show this message"
            echo "  -p, --cuda-path   Specify the path where cuda is installed (default: /usr/local/cuda-8.0)"
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            echo "Usage: $0 environment_name [OPTIONS] command [arg1..argn]" >&2
            exit 1
            ;;
    esac
done


environment="$1"
shift 1

source activate $environment

if [[ $environment =~ ^tensorflow.* ||$environment =~ ^keras.* || $environment =~ ^pytorch.* ]]; then
    export CUDA_HOME=$cuda_path
    export CUDA_ROOT=$CUDA_HOME
	export PATH=$CUDA_HOME/bin:$PATH
	export MANPATH=$CUDA_HOME/doc/man:$MANPATH
	export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi

eval "$@"

