using UnityEngine;
#if Sentis
using Unity.Sentis;
#else
using Unity.Barracuda;
#endif

namespace Mediapipe.PoseDetection{
    [CreateAssetMenu(fileName = "PoseDetection", menuName = "ScriptableObjects/Pose Detection Resource")]
    public class PoseDetectionResource : ScriptableObject
    {
        public ComputeShader preProcessCS;
        public ComputeShader postProcessCS;
        public ComputeShader postProcess2CS;
#if Sentis
        public ModelAsset model;
#else
        public NNModel model;
#endif
    }
}