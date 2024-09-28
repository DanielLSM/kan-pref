import os

docker_image = "dlsm666/kan-pref-cpu:latest"
command = "bash -c 'cd /home/mambauser/code/rl_zoo3 && bash scripts/run_walker_flora.sh'"

def run_docker(command):
    os.system(f"docker run --user root --rm {docker_image} {command}")  # Run the specified script

def main():
    run_docker(command)  # Execute the command

if __name__ == '__main__':
    main()
