Shader "ClinicalFacility/EdgeDetectionPost"
{
    Properties
    {
        _EdgeColor ("Edge Color", Color) = (0,0,0,1)
        _EdgeOpacity ("Edge Opacity", Range(0,1)) = 0.85
        _DepthSensitivity ("Depth Sensitivity", Range(0.01, 5)) = 1.0
        _NormalSensitivity ("Normal Sensitivity", Range(0.01, 5)) = 1.0
        _EdgeThickness ("Edge Thickness (px)", Range(0.5, 4)) = 1.0
    }

    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" }
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            Name "SobelEdgeDetect"

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareNormalsTexture.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"

            TEXTURE2D(_BlitTexture); SAMPLER(sampler_BlitTexture);
            float4 _BlitTexture_TexelSize;

            half4 _EdgeColor;
            half _EdgeOpacity;
            half _DepthSensitivity;
            half _NormalSensitivity;
            half _EdgeThickness;

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            Varyings Vert(uint vertexID : SV_VertexID)
            {
                Varyings OUT;
                OUT.positionCS = GetFullScreenTriangleVertexPosition(vertexID);
                OUT.uv = GetFullScreenTriangleTexCoord(vertexID);
                return OUT;
            }

            float SampleLinearDepth(float2 uv)
            {
                float raw = SampleSceneDepth(uv);
                return LinearEyeDepth(raw, _ZBufferParams);
            }

            half4 Frag(Varyings IN) : SV_Target
            {
                float2 texel = _BlitTexture_TexelSize.xy * _EdgeThickness;
                float2 uv = IN.uv;

                // 3x3 neighborhood offsets
                float2 offsets[8] = {
                    float2(-texel.x, -texel.y), float2(0, -texel.y), float2(texel.x, -texel.y),
                    float2(-texel.x,  0),                            float2(texel.x,  0),
                    float2(-texel.x,  texel.y), float2(0,  texel.y), float2(texel.x,  texel.y)
                };

                // Sobel kernels
                float sobelX[8] = { -1, 0, 1, -2, 2, -1, 0, 1 };
                float sobelY[8] = { -1, -2, -1, 0, 0, 1, 2, 1 };

                float depthGX = 0, depthGY = 0;
                half3 normGX = 0, normGY = 0;

                UNITY_UNROLL
                for (int i = 0; i < 8; i++)
                {
                    float2 sampleUV = uv + offsets[i];
                    float d = SampleLinearDepth(sampleUV);
                    half3 n = SampleSceneNormals(sampleUV);

                    depthGX += d * sobelX[i];
                    depthGY += d * sobelY[i];
                    normGX += n * sobelX[i];
                    normGY += n * sobelY[i];
                }

                float depthEdge = sqrt(depthGX * depthGX + depthGY * depthGY) * _DepthSensitivity;
                float normalEdge = length(normGX) + length(normGY);
                normalEdge *= _NormalSensitivity;

                float edge = saturate(max(depthEdge, normalEdge));

                half4 sceneColor = SAMPLE_TEXTURE2D(_BlitTexture, sampler_BlitTexture, uv);
                half3 outColor = lerp(sceneColor.rgb, _EdgeColor.rgb, edge * _EdgeOpacity);

                return half4(outColor, sceneColor.a);
            }
            ENDHLSL
        }
    }
}
