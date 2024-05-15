#ifndef POSE_DETECTION_COMMON_H_
#define POSE_DETECTION_COMMON_H_

#define MAX_DETECTION 64

struct PoseDetection
{
    
    float2 center;
    float2 extent;
    float2 keyPoints[4];
    float score;
};

float3x4  DetectionToMatrix(in PoseDetection d)
{
    return float3x4(d.center,d.extent,
        d.keyPoints[0],d.keyPoints[1],
        d.keyPoints[2], d.keyPoints[3]
        );
}

PoseDetection MatrixToDetection(float3x4 m,float score)
{
    PoseDetection d;
    d.center = m._m00_m01;
    d.extent = m._m02_m03;
    d.keyPoints[0]= m._m10_m11;
    d.keyPoints[1]= m._m12_m13;
    d.keyPoints[2]= m._m20_m21;
    d.keyPoints[3]= m._m22_m23;
    d.score = score;
    return d;
}

float2 VFlip(float2 p)
{
    return float2(p.x,1-p.y);
}

float Sigmoid(float x)
{
    return 1.0 / ( 1.0 + exp( - x ));
}

float CalculateIOU(in PoseDetection d1, in PoseDetection d2)
{
    float area0 = d1.extent.x * d1.extent.y;
    float area1 = d2.extent.x * d2.extent.y;
    float2 p0 = max(d1.center - d1.extent / 2 , d2.center - d2.extent / 2);
    float2 p1 = min(d1.center + d1.extent / 2 , d2.center + d2.extent / 2);
    float areaInner = max(0, p1.x - p0.x)*max(0 , p1.y - p0.y);
    return areaInner /  ( area0 + area1 - areaInner);
}
#endif
