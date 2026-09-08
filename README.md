# TeLeRobot


## Need to create ssl certificate 

```bash

mkdir -p ssl_cert

openssl req \
-x509 \
-newkey rsa:4096 \
-keyout ssl_cert/server.key \
-out ssl_cert/server.crt \
-days 365 \
-nodes \
-subj "/CN=localhost"
```

## Calibrate the arm 

```bash 
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=single_arm_right

```

## Connecting to the arm 

```bash
cd /workspace/teleop

telerobot 
```

## Workspace GUI to determine boundary boxes 

```bash

cd /workspace/teleop/tools 

python workspace_bounds_gui.py 
```


## Deleting Episodes
To delete an episode, run the following command with the list of episodes you want to delete(you need to be logged in for --push-to-hub to work):

```bash
poetry run telerobot delete-episodes --repo-id <YOUR_HF_USERNAME>/vr_test_single_arm> --episodes "[0, 2, 5]" --push-to-hub
```

