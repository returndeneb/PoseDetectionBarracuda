using Unity.Sentis;
using UnityEngine;
using Klak.NNUtils;
using Klak.NNUtils.Extensions;

namespace Mediapipe.PoseDetection {

public sealed class PalmDetector : System.IDisposable
{
    #region Public methods/properties

    public PalmDetector(PoseDetectionResource resources)
      => AllocateObjects(resources);

    public void Dispose()
      => DeallocateObjects();

    public void ProcessImage(Texture image, float threshold = 0.75f)
      => RunModel(image, threshold);

    public System.ReadOnlySpan<PoseDetection> Detections
      => _readCache.Cached;

    public GraphicsBuffer DetectionBuffer
      => _output.post2;

    #endregion

    #region Private objects

    PoseDetectionResource _resources;
    int _size;
    IWorker _worker;
    ImagePreprocess _preprocess;
    (GraphicsBuffer post1, GraphicsBuffer post2, GraphicsBuffer count) _output;
    CountedBufferReader<PoseDetection> _readCache;

    void AllocateObjects(PoseDetectionResource resources)
    {
        _resources = resources;

        // NN model
        var model = ModelLoader.Load(_resources.model);
        _size = model.inputs[0].GetTensorShape().GetWidth();

        // GPU worker
        _worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
        // _worker = model.CreateWorker(WorkerFactory.Device.GPU);

        // Preprocess
        _preprocess = new ImagePreprocess(_size, _size, nchwFix: true);

        // Output buffers
        _output.post1 = BufferUtil.NewAppend<PoseDetection>(PoseDetection.Max);
        _output.post2 = BufferUtil.NewAppend<PoseDetection>(PoseDetection.Max);
        _output.count = BufferUtil.NewRaw(1);

        // Detection data read cache
        _readCache = new CountedBufferReader<PoseDetection>(_output.post2, _output.count, PoseDetection.Max);
    }

    void DeallocateObjects()
    {
        _worker?.Dispose();
        _worker = null;

        _preprocess?.Dispose();
        _preprocess = null;

        _output.post1?.Dispose();
        _output.post2?.Dispose();
        _output.count?.Dispose();
        _output = (null, null, null);
    }

    #endregion

    #region Neural network inference function

    void RunModel(Texture source, float threshold)
    {
        _preprocess.Dispatch(source, _resources.preProcessCS);
        RunModel(threshold);
    }

    void RunModel(float threshold)
    {
        // Reset the compute buffer counters.
        _output.post1.SetCounterValue(0);
        _output.post2.SetCounterValue(0);

        // Run the BlazePalm model.
        _worker.Execute(_preprocess.Tensor);

        // 1st postprocess (bounding box aggregation)
        var post1 = _resources.postProcessCS;
        post1.SetFloat("_ImageSize", _size);
        post1.SetFloat("_Threshold", threshold);
        // var scores = _worker.PeekOutputBuffer("classificators");
        // Debug.Log(scores.count);
        post1.SetBuffer(0, "_Scores", _worker.PeekOutputBuffer("classificators"));
        post1.SetBuffer(0, "_Boxes", _worker.PeekOutputBuffer("regressors"));
        post1.SetBuffer(0, "_Output", _output.post1);
        post1.Dispatch(0, 1, 1, 1);

        post1.SetBuffer(1, "_Scores", _worker.PeekOutputBuffer("classificators"));
        post1.SetBuffer(1, "_Boxes", _worker.PeekOutputBuffer("regressors"));
        post1.SetBuffer(1, "_Output", _output.post1);
        post1.Dispatch(1, 1, 1, 1);

        // Retrieve the bounding box count.
        GraphicsBuffer.CopyCount(_output.post1, _output.count, 0);

        // 2nd postprocess (overlap removal)
        var post2 = _resources.postProcess2CS;
        post2.SetBuffer(0, "_inputBuffer", _output.post1);
        post2.SetBuffer(0, "_inputCountBuffer", _output.count);
        post2.SetBuffer(0, "_output", _output.post2);
        post2.Dispatch(0, 1, 1, 1);

        // Retrieve the bounding box count after removal.
        GraphicsBuffer.CopyCount(_output.post2, _output.count, 0);

        // Cache data invalidation
        _readCache.InvalidateCache();
    }

    #endregion
}

} // namespace MediaPipe.BlazePalm
