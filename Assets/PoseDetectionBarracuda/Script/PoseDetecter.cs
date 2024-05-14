using UnityEngine;
#if Sentis
using Unity.Sentis;
#else
using Unity.Barracuda;
#endif


namespace Mediapipe.PoseDetection{
    public class PoseDetecter: System.IDisposable
    {
        #region public variable
        // Pose detection result buffer.
        public ComputeBuffer outputBuffer;
        // Pose detection result count buffer.
        public ComputeBuffer countBuffer;
        #endregion

        #region constant number 
        // Input image size defined by pose detection network model.
        const int IMAGE_SIZE = 128;
        // MAX_DETECTION must be matched with "Postprocess2.compute"
        const int MAX_DETECTION = 64;
        #endregion

        #region private variable
        IWorker woker;
        Model model;
        ComputeShader preProcessCS;
        ComputeShader postProcessCS;
        ComputeShader postProcess2CS;
        ComputeBuffer postProcessBuffer;
        // ComputeBuffer networkInputBuffer;

        private ComputeTensorData data;
#if Sentis
        TensorShape shape =new (1,3,IMAGE_SIZE, IMAGE_SIZE);
#else
        TensorShape shape =new (1,IMAGE_SIZE, IMAGE_SIZE, 3);
#endif
        private Tensor tensor;
        
        #endregion

        #region public method
        public PoseDetecter(PoseDetectionResource resource){
            preProcessCS = resource.preProcessCS;
            postProcessCS = resource.postProcessCS;
            postProcess2CS = resource.postProcess2CS;
            
            outputBuffer = new ComputeBuffer(MAX_DETECTION, PoseDetection.Size, ComputeBufferType.Append);
            countBuffer = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Raw);
            // networkInputBuffer = new ComputeBuffer(IMAGE_SIZE * IMAGE_SIZE * 3, sizeof(float));
            
            model = ModelLoader.Load(resource.model);
            postProcessBuffer = new ComputeBuffer(MAX_DETECTION, PoseDetection.Size, ComputeBufferType.Append);
#if Sentis
            data = new ComputeTensorData(shape,false);
            tensor = TensorFloat.Zeros(shape);
            woker = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
#else
            data = new ComputeTensorData(shape, "preprocess", ComputeInfo.ChannelsOrder.NHWC, false);
            woker = model.CreateWorker(WorkerFactory.Device.GPU);
#endif
        
        }

        public void Dispose(){
            outputBuffer.Dispose();
            countBuffer.Dispose();
            // networkInputBuffer.Dispose();
            data=null;
            postProcessBuffer.Dispose();
            woker.Dispose();
        }

        public void ProcessImage(Texture inputTexture, float poseThreshold = 0.75f, float iouThreshold = 0.3f){
            // Resize `inputTexture` texture to network model image size.
            preProcessCS.SetTexture(0, "_inputTexture", inputTexture);
            preProcessCS.SetBuffer(0, "_output", data.buffer);
            preProcessCS.Dispatch(0, IMAGE_SIZE / 8, IMAGE_SIZE / 8, 1);

            // Reset append type buffer datas of previous frame. 
            postProcessBuffer.SetCounterValue(0);
            outputBuffer.SetCounterValue(0);
#if Sentis
            tensor.AttachToDevice(data);
#else
            tensor = new Tensor(shape, data);
#endif
            
            //Execute neural network model.
            woker.Execute(tensor);

            //Get neural network model raw output as RenderTexture;
            // var scores = CopyOutputToTempRT("classificators", 1, 896);
            var scores = ((ComputeTensorData)woker.PeekOutput("classificators").tensorOnDevice).buffer;
            // var boxs = CopyOutputToTempRT("regressors", 12, 896);
            var boxs = ((ComputeTensorData)woker.PeekOutput("regressors").tensorOnDevice).buffer;
            
            // Parse raw result datas for above values of vectors.
            postProcessCS.SetFloat("_threshold", poseThreshold);
            postProcessCS.SetBuffer(0, "_scores", scores);
            postProcessCS.SetBuffer(0, "_boxs", boxs);
            postProcessCS.SetBuffer(0, "_output", postProcessBuffer);
            postProcessCS.Dispatch(0, 1, 1, 1);

            // Parse raw result datas for behind values of vectors.
            postProcessCS.SetBuffer(1, "_scores", scores);
            postProcessCS.SetBuffer(1, "_boxs", boxs);
            postProcessCS.SetBuffer(1, "_output", postProcessBuffer);
            postProcessCS.Dispatch(1, 1, 1, 1);

            // RenderTexture.ReleaseTemporary(scores);
            // RenderTexture.ReleaseTemporary(boxs);
            ComputeBuffer.CopyCount(postProcessBuffer, countBuffer, 0);
            
            // Get final results of pose deteciton.
            postProcess2CS.SetFloat("_iouThreshold", iouThreshold);
            postProcess2CS.SetBuffer(0, "_inputBuffer", postProcessBuffer);
            postProcess2CS.SetBuffer(0, "_inputCountBuffer", countBuffer);
            postProcess2CS.SetBuffer(0, "_output", outputBuffer);
            postProcess2CS.Dispatch(0, 1, 1, 1);

            // Set pose detection results count.
            ComputeBuffer.CopyCount(outputBuffer, countBuffer, 0);
        }

        #endregion

    }
}
