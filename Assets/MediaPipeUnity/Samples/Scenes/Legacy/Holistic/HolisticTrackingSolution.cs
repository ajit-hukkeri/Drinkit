using System.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using TMPro;
using Mediapipe.Unity.Sample.ObjectDetection;
using cup;

namespace Mediapipe.Unity.Sample.Holistic
{
    
    public class HolisticTrackingSolution : LegacySolutionRunner<HolisticTrackingGraph>
    {
        [SerializeField] private RectTransform _worldAnnotationArea;
        [SerializeField] private DetectionAnnotationController _poseDetectionAnnotationController;
        [SerializeField] private HolisticLandmarkListAnnotationController _holisticAnnotationController;
        [SerializeField] private PoseWorldLandmarkListAnnotationController _poseWorldLandmarksAnnotationController;
        [SerializeField] private MaskAnnotationController _segmentationMaskAnnotationController;
        [SerializeField] private NormalizedRectAnnotationController _poseRoiAnnotationController;
        [SerializeField] private TMP_Text messageText; 

        private Experimental.TextureFramePool _textureFramePool;

        public HolisticTrackingGraph.ModelComplexity modelComplexity
        {
            get => graphRunner.modelComplexity;
            set => graphRunner.modelComplexity = value;
        }

        public bool smoothLandmarks
        {
            get => graphRunner.smoothLandmarks;
            set => graphRunner.smoothLandmarks = value;
        }

        public bool refineFaceLandmarks
        {
            get => graphRunner.refineFaceLandmarks;
            set => graphRunner.refineFaceLandmarks = value;
        }

        public bool enableSegmentation
        {
            get => graphRunner.enableSegmentation;
            set => graphRunner.enableSegmentation = value;
        }

        public bool smoothSegmentation
        {
            get => graphRunner.smoothSegmentation;
            set => graphRunner.smoothSegmentation = value;
        }

        public float minDetectionConfidence
        {
            get => graphRunner.minDetectionConfidence;
            set => graphRunner.minDetectionConfidence = value;
        }

        public float minTrackingConfidence
        {
            get => graphRunner.minTrackingConfidence;
            set => graphRunner.minTrackingConfidence = value;
        }

        private float CalculateDistance(NormalizedLandmark landmark1, NormalizedLandmark landmark2)
        {
            float xDiff = landmark1.X - landmark2.X;
            float yDiff = landmark1.Y - landmark2.Y;
            return Mathf.Sqrt(xDiff * xDiff + yDiff * yDiff);
        }

        
        private bool IsHandNearMouth(NormalizedLandmarkList handLandmarks, NormalizedLandmarkList faceLandmarks)
        {
            if (handLandmarks == null || faceLandmarks == null)
            {
                return false;
            }

            
            var mouth = faceLandmarks.Landmark[13];
            var handTip = handLandmarks.Landmark[4]; 

            
            float distance = CalculateDistance(mouth, handTip);

           
            return distance < 0.13f;
        }
        public Vector2 cupPos = cupdata.cupPosition;
        private bool IsCupNearMouth(Vector2 cupPos, NormalizedLandmarkList faceLandmarks)
        {
            if (cupPos == null || faceLandmarks == null)
            {
                return false;
            }

            
            var mouth = faceLandmarks.Landmark[13];

            
            float distance = Vector2.Distance(cupPos, new Vector2(mouth.X, mouth.Y));
            Debug.Log(distance);

            
            return distance < 0.8; 
        }
        protected override IEnumerator Run()
        {
            var graphInitRequest = graphRunner.WaitForInit(runningMode);
            var imageSource = ImageSourceProvider.ImageSource;

            yield return imageSource.Play();

            if (!imageSource.isPrepared)
            {
                Debug.LogError("Failed to start ImageSource, exiting...");
                yield break;
            }

            _textureFramePool = new Experimental.TextureFramePool(imageSource.textureWidth, imageSource.textureHeight, TextureFormat.RGBA32, 10);
            screen.Initialize(imageSource);
            _worldAnnotationArea.localEulerAngles = imageSource.rotation.Reverse().GetEulerAngles();

            yield return graphInitRequest;
            if (graphInitRequest.isError)
            {
                Debug.LogError(graphInitRequest.error);
                yield break;
            }

            if (!runningMode.IsSynchronous())
            {
                graphRunner.OnPoseDetectionOutput += OnPoseDetectionOutput;
                graphRunner.OnFaceLandmarksOutput += OnFaceLandmarksOutput;
                graphRunner.OnPoseLandmarksOutput += OnPoseLandmarksOutput;
                graphRunner.OnLeftHandLandmarksOutput += OnLeftHandLandmarksOutput;
                graphRunner.OnRightHandLandmarksOutput += OnRightHandLandmarksOutput;
                graphRunner.OnPoseWorldLandmarksOutput += OnPoseWorldLandmarksOutput;
                graphRunner.OnSegmentationMaskOutput += OnSegmentationMaskOutput;
                graphRunner.OnPoseRoiOutput += OnPoseRoiOutput;
            }

            SetupAnnotationController(_poseDetectionAnnotationController, imageSource);
            SetupAnnotationController(_holisticAnnotationController, imageSource);
            SetupAnnotationController(_poseWorldLandmarksAnnotationController, imageSource);
            SetupAnnotationController(_segmentationMaskAnnotationController, imageSource);
            _segmentationMaskAnnotationController.InitScreen(imageSource.textureWidth, imageSource.textureHeight);
            SetupAnnotationController(_poseRoiAnnotationController, imageSource);

            graphRunner.StartRun(imageSource);

            AsyncGPUReadbackRequest req = default;
            var waitUntilReqDone = new WaitUntil(() => req.done);
            var canUseGpuImage = graphRunner.configType == GraphRunner.ConfigType.OpenGLES && GpuManager.GpuResources != null;
            using var glContext = canUseGpuImage ? GpuManager.GetGlContext() : null;

            while (true)
            {
                if (isPaused)
                {
                    yield return new WaitWhile(() => isPaused);
                }

                if (!_textureFramePool.TryGetTextureFrame(out var textureFrame))
                {
                    yield return new WaitForEndOfFrame();
                    continue;
                }

                if (canUseGpuImage)
                {
                    yield return new WaitForEndOfFrame();
                    textureFrame.ReadTextureOnGPU(imageSource.GetCurrentTexture());
                }
                else
                {
                    req = textureFrame.ReadTextureAsync(imageSource.GetCurrentTexture());
                    yield return waitUntilReqDone;

                    if (req.hasError)
                    {
                        Debug.LogError($"Failed to read texture from the image source, exiting...");
                        break;
                    }
                }

                graphRunner.AddTextureFrameToInputStream(textureFrame, glContext);

                if (runningMode.IsSynchronous())
                {
                    screen.ReadSync(textureFrame);

                    var task = graphRunner.WaitNextAsync();
                    yield return new WaitUntil(() => task.IsCompleted);

                    var result = task.Result;
                    _poseDetectionAnnotationController.DrawNow(result.poseDetection);
                    _holisticAnnotationController.DrawNow(result.faceLandmarks, result.poseLandmarks, result.leftHandLandmarks, result.rightHandLandmarks);
                    _poseWorldLandmarksAnnotationController.DrawNow(result.poseWorldLandmarks);
                    _segmentationMaskAnnotationController.DrawNow(result.segmentationMask);
                    _poseRoiAnnotationController.DrawNow(result.poseRoi);

                    result.segmentationMask?.Dispose();

                    
                    CheckForDrinking(result.leftHandLandmarks, result.rightHandLandmarks, result.faceLandmarks,cup.cupdata.cupPosition);
                }
            }
        }

        private void OnPoseDetectionOutput(object stream, OutputStream<Detection>.OutputEventArgs eventArgs)
        {
            var packet = eventArgs.packet;
            var value = packet == null ? default : packet.Get(Detection.Parser);
            _poseDetectionAnnotationController.DrawLater(value);
        }

        private void OnFaceLandmarksOutput(object stream, OutputStream<NormalizedLandmarkList>.OutputEventArgs eventArgs)
        {
            var packet = eventArgs.packet;
            var value = packet == null ? default : packet.Get(NormalizedLandmarkList.Parser);
            _holisticAnnotationController.DrawFaceLandmarkListLater(value);
        }

        private void OnPoseLandmarksOutput(object stream, OutputStream<NormalizedLandmarkList>.OutputEventArgs eventArgs)
        {
            var packet = eventArgs.packet;
            var value = packet == null ? default : packet.Get(NormalizedLandmarkList.Parser);
            _holisticAnnotationController.DrawPoseLandmarkListLater(value);
        }

        private void OnLeftHandLandmarksOutput(object stream, OutputStream<NormalizedLandmarkList>.OutputEventArgs eventArgs)
        {
            var packet = eventArgs.packet;
            var value = packet == null ? default : packet.Get(NormalizedLandmarkList.Parser);
            _holisticAnnotationController.DrawLeftHandLandmarkListLater(value);
        }

        private void OnRightHandLandmarksOutput(object stream, OutputStream<NormalizedLandmarkList>.OutputEventArgs eventArgs)
        {
            var packet = eventArgs.packet;
            var value = packet == null ? default : packet.Get(NormalizedLandmarkList.Parser);
            _holisticAnnotationController.DrawRightHandLandmarkListLater(value);
        }

        private void OnPoseWorldLandmarksOutput(object stream, OutputStream<LandmarkList>.OutputEventArgs eventArgs)
        {
            var packet = eventArgs.packet;
            var value = packet == null ? default : packet.Get(LandmarkList.Parser);
            _poseWorldLandmarksAnnotationController.DrawLater(value);
        }

        private void OnSegmentationMaskOutput(object stream, OutputStream<ImageFrame>.OutputEventArgs eventArgs)
        {
            var packet = eventArgs.packet;
            var value = packet == null ? default : packet.Get();
            _segmentationMaskAnnotationController.DrawLater(value);
            value?.Dispose();
        }

        private void OnPoseRoiOutput(object stream, OutputStream<NormalizedRect>.OutputEventArgs eventArgs)
        {
            var packet = eventArgs.packet;
            var value = packet == null ? default : packet.Get(NormalizedRect.Parser);
            _poseRoiAnnotationController.DrawLater(value);
        }

        
        private void CheckForDrinking(NormalizedLandmarkList leftHandLandmarks, NormalizedLandmarkList rightHandLandmarks, NormalizedLandmarkList faceLandmarks, Vector2 cupPosition)
        {
            bool isDrinkingDetected = false;

            
            if (faceLandmarks != null)
            {
                
                bool handNearMouth = false;

                if (leftHandLandmarks != null)
                {
                    var leftHand = leftHandLandmarks.Landmark[6];
                    float distanceLeft = CalculateDistance(leftHand, faceLandmarks.Landmark[13]);  // 13 is the mouth landmark
                    if (distanceLeft < 0.13f)
                    {
                        handNearMouth = true;
                    }
                }

                if (rightHandLandmarks != null)
                {
                    var rightHand = rightHandLandmarks.Landmark[6];
                    float distanceRight = CalculateDistance(rightHand, faceLandmarks.Landmark[13]);  
                    if (distanceRight < 0.13f)
                    {
                        handNearMouth = true;
                    }
                }

                
                bool cupNearMouth = IsCupNearMouth(cupPosition, faceLandmarks);

               
                if (handNearMouth && cupNearMouth)
                {
                    isDrinkingDetected = true;
                }
            }

            
            if (isDrinkingDetected)
            {
                DisplayMessage("Drinking Detected!");
            }
            else
            {
                DisplayMessage("Waiting for drinking detection");
            }
        }


        private void DisplayMessage(string message)
        {
            if (messageText != null)
            {
                messageText.text = message;
            }
        }

    }
}
