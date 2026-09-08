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

## Git Pull

```bash
git pull origin main
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


## Setting up hugging face cli to upload datasets 

> Create hugging face account/ log in

> Click on your profile and click on access tocken 

> Create new access token name it 'rif token' then choose permissions for 'write' or 'full access' 

> Save your token locally 

> To create a dataset, click on your profile and click ' new data set' 

> for the dataset name you can use 'vr_test_single_arm'

> choose public 

> click 'create dataset 

> then follow the auth hf login instructions, you will your access token here 

> One the auth hf is setup then, go to 'repo_id' in the config yaml file and update the 'hf user id/data set name'





## Deleting Episodes
To delete an episode, run the following command with the list of episodes you want to delete(you need to be logged in for --push-to-hub to work):

```bash
poetry run telerobot delete-episodes --repo-id <YOUR_HF_USERNAME>/vr_test_single_arm> --episodes "[0, 2, 5]" --push-to-hub

#Delte local dataset

find /workspace/.cache/lerobot -type d -path '*vr_test_single_arm*'

rm -rf <path the cmd returned>
```



