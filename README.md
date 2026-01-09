# virtual_zebrafish

We don't use Unity anymore:

<s>
## Unity Setup
0. First, install the Mujoco Unity application for your specific platform that you will develop on (in my case, it is Mac OS X), found [here](https://github.com/google-deepmind/mujoco/releases). For Mac, open the DMG file, copy the `Mujoco` app over to `Applications/`, and open it once.
1. Then, download the source (`.zip` file) at the release same version as Step 0, e.g. for 2.3.7 it is [here](https://github.com/google-deepmind/mujoco/archive/refs/tags/2.3.7.zip). Unzip the zip file, and then add `package.json` [taken from the Unity subdirectory](https://github.com/deepmind/mujoco/tree/main/unity) using the Unity Package Manager (`Window->Package Manager`) via the "Add from local disk" option.
2. Install Unity ML Agents via the package manager by selecting packages from the Unity Registry.
3. In Python, generate an XML file with `n_joints` that you specify (must be at least 3 -- this example is 3 joints):
```
from dm_control.suite.swimmer import get_model_and_assets
model_string, _ = get_model_and_assets(3)
with open("swimmer3.xml", "w") as f:
    f.write(model_string.decode())
```
4. Copy (or write directly) the resultant `.xml` file to the `Assets/` directory of your Unity project, and copy the `common/` folder there too (from the [DM Control Suite](https://github.com/deepmind/dm_control/tree/main/dm_control/suite/common)). You need the `common/` folder for the materials, colors, etc.
5. In Unity, select `Assets->Import MuJoCo Scene` and select the `.xml` file. Viola!
6. Rename the imported name of the agent to `swimmer3`. You will then have to organize the assets as in `SampleScene` and make the target a Prefab in order to move it to a random location each time. *Also, remove `sensors/target_pos` if you do turn the target into a Prefab; otherwise it will crash (since the target object has been deleted in the hierarchy).*
7. Write your agent script and save it to the `Assets/Scripts` folder. Once you're done with it, select the top-level `swimmer3` GameObject and attach it to there.
8. Once you do that, it will automatically give a `Behavior Parameters` component. Modify the following fields: Set `Vector Observation->Space Size` to be match the number of observations (8). Then set `Actions->Continuous Actions` to match the number of actuators (2), and set `Actions->Discrete Branches` to 0. Finally, attach a `Camera Sensor` component, and set the camera to be `eyes`, to provide pixel observations later on (we will ignore this for now, but the RL environment won't train unless there are those two types of inputs). You can optionally add a `DecisionRequestor` component, and if you have that set in your Agent script, you will see that value change to the preset script value once you hit Play. Set the main camera Target Display to be Display 1, and then the other cameras to have some other Target Displays (not Display 1).
9. To move the agent around in Unity with your mouse, enter play mode in Unity (click play button at the top of the screen) and then enter Scene mode _while playing_. Then select the "head" object in the Swimmer3 hierarchy (under "target"), then hold down ctrl + shift while left clicking on the agent. You can now control the agent with your mouse.
10. Once you have the scene and you want to run it on the OpenMind Linux cluster, create a folder with the convention: `{name_of_env}_unity`. Then, build it under `Build Settings->Dedicated Server` with the `Target Platform` set to `Linux`, with the name `swimmer3`. **This is only because we don't have graphics (for now)!** If you don't see this option, then you need to install the `Linux Dedicated Module` as an added module to the Unity Editor in `UnityHub`. Note that the first time you select the option, it can take a while for the `Build And Run` option to be selectable because it is compiling the compute variants, and you can see the progress in the bottom right. If you don't see it be selectable, hit `Switch Platform`, especially if doing this the first time.
11. Then, add Mujoco to the Linux build by extracting [this](https://github.com/deepmind/mujoco/releases/download/2.3.7/mujoco-2.3.7-linux-x86_64.tar.gz), and copy over `mujoco/lib/libmujoco.so.2.3.7` as `libmujoco.so` to the `swimmer3_Data/MonoBleedingEdge/x86_64/` folder.
 
## ANTS Installation for Segmentation of Volumes (likely not needed)
- ANTS (Advanced Normalization Tools) should already be installed at `/om2/group/yanglab/ants/install/bin/`.
- But if you need to install ANTS, you need 4 threads for install:
    ```
    cp installANTS.sh /om2/your/directory
    cd /om2/your/directory/
    srun -n 4 -t 168:00:00 -p yanglab --mem=50G --pty bash
    chmod a+x installANTS.sh
    ./installANTS.sh
    ```
   This script is taken from: https://github.com/cookpa/antsInstallExample
</s>
