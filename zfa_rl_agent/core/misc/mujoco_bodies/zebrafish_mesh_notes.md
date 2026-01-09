# Giving Mesh Bones in Blender

These notes will help you create a skin for your MuJoCo agent, which is assumed to be running in Unity. You will need to have both Unity and Blender installed, in addition to following the steps in the main repo for hooking MuJoco and Unity together.

Here are the steps one can follow:

- Step 1: Create a custom mesh in Blender or import a prexisting one (for example, after purchasing it online). I bought and used [this one](https://skfb.ly/6oz98).
- Step 2: Create armature for the mesh in Blender. The number of armature links should be equal to the number of links in the swimmer agent. To learn how to perform this step, there are many resources online. I found YouTube videos to be the most helpful, for example [this one](https://www.youtube.com/watch?v=9dZjcFW3BRY&t=508s).
- Step 3: Export the mesh and armature as an .fbx file, and then import that .fbx file into Unity as an asset (right click in the asset folder and select "Import New Asset").
- Step 4: Bring the imported asset into your Unity scene by clicking and dragging it into the scene hierarchy (on the left side of the screen).
- Step 5: Right click on the asset, and select "Prefab", and then select "Unpack Prefab Completely".
- Step 6: Manually assign each of the mesh armatures to the corresponding link in the MuJoCo agent by clicking and dragging it to the appropriate place in the MuJoCo agent hierarchy. For example, the mesh head armature should go under the visual component of the agent head, the mesh torso link (if the agent is a swimmer) should go under the visual component of segment_0 in the agent, etc. See screenshot below, where I've named the mesh components to correspond to the MuJoCo agent components. You can check that the mesh is correctly hooked up by manually rotating and moving the MuJoco agent links. The mesh should be driven by them.

![image](https://github.com/kozleo/virtual_zebrafish/assets/20054919/fc2bb1de-5874-4246-8230-a38fbe1230ea)
