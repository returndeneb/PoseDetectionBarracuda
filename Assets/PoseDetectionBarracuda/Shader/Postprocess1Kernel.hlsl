// #include "Postprocess1.compute"

[numthreads(mapSize, mapSize, 1)]
void KERNEL_NAME(uint2 id: SV_DispatchThreadID)
{
    float scale = 1.0 / _ImageSize;
    float2 anchor = (0.5 + id) / mapSize;
    uint index_In0chMap = (id.y * mapSize + id.x) * chSize + indexOffset;
    uint bidx = index_In0chMap *12;
    for(uint i=0; i<chSize; i++){
        
        PoseDetection pd;
        pd.score = Sigmoid(_Scores[index_In0chMap++]);
        
        float x = _Boxes[bidx++] ;
        float y = _Boxes[bidx++] ;
        float w = _Boxes[bidx++] ;
        float h = _Boxes[bidx++] ;
        pd.center = anchor + float2(x, y)* scale;
        pd.extent = float2(w, h)* scale;
        
        [unroll] for(uint i=0; i<4; i++){
            x = _Boxes[bidx++] ;
            y = _Boxes[bidx++] ;
            pd.keyPoints[i] = anchor + float2(x, y)* scale;
        }

        if (pd.score > _Threshold)_Output.Append(pd);
    }
}
