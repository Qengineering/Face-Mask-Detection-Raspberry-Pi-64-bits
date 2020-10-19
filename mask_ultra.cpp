// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <string>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include "paddle_api.h"  // NOLINT
#include "UltraFace.hpp"

using namespace std;
using namespace paddle::lite_api;  // NOLINT

int main(int argc,char ** argv)
{
    float f;
    float FPS[16];
    int i, Fcnt=0;
    cv::Mat frame;
    int classify_w = 128;
    int classify_h = 128;
    float scale_factor = 1.f / 256;
    int FaceImgSz  = classify_w * classify_h;

    // Mask detection (second phase, when the faces are located)
    MobileConfig Mconfig;
    std::shared_ptr<PaddlePredictor> Mpredictor;
    //some timing
    chrono::steady_clock::time_point Tbegin, Tend;

    for(i=0;i<16;i++) FPS[i]=0.0;

    //load SSD face detection model and get predictor
//    UltraFace ultraface("slim_320.bin","slim_320.param", 320, 240, 2, 0.7); // config model input
    UltraFace ultraface("RFB-320.bin","RFB-320.param", 320, 240, 2, 0.7); // config model input

    //load mask detection model
    Mconfig.set_model_from_file("mask_detector_opt2.nb");
    Mpredictor = CreatePaddlePredictor<MobileConfig>(Mconfig);
    std::cout << "Load classification model succeed." << std::endl;

    // Get Input Tensor
    std::unique_ptr<Tensor> input_tensor1(std::move(Mpredictor->GetInput(0)));
    input_tensor1->Resize({1, 3, classify_h, classify_w});

    // Get Output Tensor
    std::unique_ptr<const Tensor> output_tensor1(std::move(Mpredictor->GetOutput(0)));

    cv::VideoCapture cap("Face_Mask_Video.mp4");
    if (!cap.isOpened()) {
        cerr << "ERROR: Unable to open the camera" << endl;
        return 0;
    }
    cout << "Start grabbing, press ESC on Live window to terminate" << endl;

    while(1){
//        frame=cv::imread("Face_2.jpg");  //if you want to run just one picture need to refresh frame before class detection
        cap >> frame;
        if (frame.empty()) {
            cerr << "ERROR: Unable to grab from the camera" << endl;
            break;
        }

        Tbegin = chrono::steady_clock::now();

        ncnn::Mat inmat = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);

        //get the faces
        std::vector<FaceInfo> face_info;
        ultraface.detect(inmat, face_info);

        auto* input_data = input_tensor1->mutable_data<float>();

        for(long unsigned int i = 0; i < face_info.size(); i++) {
            auto face = face_info[i];
            //enlarge 10%
            float w = (face.x2 - face.x1)/20.0;
            float h = (face.y2 - face.y1)/20.0;
            cv::Point pt1(std::max(face.x1-w,float(0.0)),std::max(face.y1-h,float(0.0)));
            cv::Point pt2(std::min(face.x2+w,float(frame.cols)),std::min(face.y2+h,float(frame.rows)));
            //RecClip is completly inside the frame
            cv::Rect  RecClip(pt1, pt2);
            cv::Mat   resized_img;
            cv::Mat   imgf;

            if(RecClip.width>0 && RecClip.height>0){
                //roi has size RecClip
                cv::Mat roi = frame(RecClip);

                //resized_img has size 128x128 (uchar)
                cv::resize(roi, resized_img, cv::Size(classify_w, classify_h), 0.f, 0.f, cv::INTER_CUBIC);

                //imgf has size 128x128 (float in range 0.0 - +1.0)
                resized_img.convertTo(imgf, CV_32FC3, scale_factor);

                //input tensor has size 128x128 (float in range -0.5 - +0.5)
                // fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
                //offset_nchw(n, c, h, w) = n * CHW + c * HW + h * W + w
                //offset_nhwc(n, c, h, w) = n * HWC + h * WC + w * C + c
                const float* dimg = reinterpret_cast<const float*>(imgf.data);

                float* dout_c0 = input_data;
                float* dout_c1 = input_data + FaceImgSz;
                float* dout_c2 = input_data + FaceImgSz * 2;

                for(int i=0;i<FaceImgSz;i++){
                    *(dout_c0++) = (*(dimg++) - 0.5);
                    *(dout_c1++) = (*(dimg++) - 0.5);
                    *(dout_c2++) = (*(dimg++) - 0.5);
                }

                // Classification Model Run
                Mpredictor->Run();

                auto* outptr = output_tensor1->data<float>();
                float prob = outptr[1];

                // Draw Detection and Classification Results
                bool flag_mask = prob > 0.5f;
                cv::Scalar roi_color;

                if(flag_mask) roi_color = cv::Scalar(0, 255, 0);
                else          roi_color = cv::Scalar(0, 0, 255);
                // Draw roi object
                cv::rectangle(frame, RecClip, roi_color, 2);
            }
        }

        Tend = chrono::steady_clock::now();
        //calculate frame rate
        f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
        if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        cv::putText(frame, cv::format("FPS %0.2f", f/16),cv::Point(10,20),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(0, 0, 255));

        //cv::imwrite("FaceResult.jpg",frame); //in case you run only a jpg picture

        //show output
        cv::imshow("RPi 64 OS - 1,95 GHz - 2 Mb RAM", frame);

        char esc = cv::waitKey(5);
        if(esc == 27) break;
    }

    cout << "Closing the camera" << endl;
    cv::destroyAllWindows();
    cout << "Bye!" << endl;

  return 0;
}
