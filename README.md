# TeLeRobot


## Need to create ssl certificate 

mkdir -p ssl_cert

openssl req \
-x509 \
-newkey rsa:4096 \
-keyout ssl_cert/server.key \
-out ssl_cert/server.crt \
-days 365 \
-nodes \
-subj "/CN=localhost"


## Connecting to the arm 

cd /workspace/teleop

telerobot 


## Workspace GUI to determine boundary boxes 

cd /workspace/teleop/tools 

python workspace_bounds_gui.py 





## Dataset Recording

To record an episode, press the Record button. Perform the desired task with the robot, then press the Save Episode button to save the episode to a LeRobot dataset and repeat for the next episode. After you record all the episodes you need, you can press the Save Dataset button to save the dataset and optionally push it to Hugging Face Hub if you enabled that feature in the config.


https://github.com/user-attachments/assets/0fe03e12-4e4c-484b-948e-5e66431d1ead


## Disabling Passthrough

To record good episodes it can be helpful to see only what the robot sees without the passthrough video from the headset. To do this, you can disable the Passthrough toggle in the web interface to hide the passthrough feed and only see the robot's camera feeds. It will be harder to controll the robot this way, but you should get better results during training.


https://github.com/user-attachments/assets/31be1a9b-55e2-443f-bd87-ab5510739faa


## Deleting Episodes
To delete an episode, run the following command with the list of episodes you want to delete(you need to be logged in for --push-to-hub to work):

```bash
poetry run telerobot delete-episodes --repo-id <YOUR_HF_USERNAME>/vr_test_single_arm> --episodes "[0, 2, 5]" --push-to-hub
```

