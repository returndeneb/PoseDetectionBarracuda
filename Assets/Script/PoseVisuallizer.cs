using UnityEngine;
using Mediapipe.PoseDetection;

public class PoseVisuallizer : MonoBehaviour
{
    [SerializeField] WebCamInput webCamInput;
    [SerializeField] Shader shader;
    [SerializeField] PoseDetectionResource poseDetectionResource;
    [SerializeField] PoseDetectionResource poseDetectionResource2;

    Material material;
    // PoseDetecter detecter;
    PalmDetector detecter2;
    ComputeBuffer boxDrawArgs;
    ComputeBuffer lineDrawArgs;

    void Start(){
        material = new Material(shader);

        var cbType = ComputeBufferType.IndirectArguments;
        boxDrawArgs = new ComputeBuffer(4, sizeof(uint), cbType);
        boxDrawArgs.SetData(new [] {3*2, 0, 0, 0});
        lineDrawArgs = new ComputeBuffer(4, sizeof(uint), cbType);
        lineDrawArgs.SetData(new [] {2, 0, 0, 0});

        // detecter = new PoseDetecter(poseDetectionResource);
        detecter2 = new PalmDetector(poseDetectionResource2);
    }

    void LateUpdate(){
        // Predict pose detection by neural network model.
        // detecter.ProcessImage(webCamInput.inputImageTexture);
        detecter2.ProcessImage(webCamInput.inputImageTexture);
    } 

    void OnRenderObject(){
        // Get predicted pose detection result.
        // material.SetBuffer("_pds", detecter.outputBuffer);
        material.SetBuffer("_pds", detecter2.DetectionBuffer);

        // Set pose detection count as vertex shader instance count.
        // GraphicsBuffer.CopyCount(detecter.outputBuffer, boxDrawArgs, sizeof(uint));
        GraphicsBuffer.CopyCount(detecter2.DetectionBuffer, boxDrawArgs, sizeof(uint));
        // GraphicsBuffer.CopyCount(detecter.outputBuffer, lineDrawArgs, sizeof(uint));
        GraphicsBuffer.CopyCount(detecter2.DetectionBuffer, lineDrawArgs, sizeof(uint));

        // Draw face region box.
        material.SetPass(0);
        Graphics.DrawProceduralIndirectNow(MeshTopology.Triangles, boxDrawArgs);

        // Draw pose region box.
        material.SetPass(1);
        Graphics.DrawProceduralIndirectNow(MeshTopology.Triangles, boxDrawArgs);

        // Draw Hip-Shoulder body line.
        material.SetPass(2);
        Graphics.DrawProceduralIndirectNow(MeshTopology.Lines, lineDrawArgs);
    }

    void OnApplicationQuit(){
        // detecter.Dispose();
        detecter2.Dispose();
        boxDrawArgs.Dispose();
        lineDrawArgs.Dispose();
    }
}
