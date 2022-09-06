// Import dependencies
import React, { useRef, useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import * as cocossd from "@tensorflow-models/coco-ssd";
import Webcam from "react-webcam";
import "./App.css";
import { drawRect } from "./utilities";
import { rand } from "@tensorflow/tfjs";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  // Main function
  const runCoco = async () => {
    const net = await cocossd.load();
    console.log("Handpose model loaded.");
    //  Loop and detect hands
    setInterval(() => {
      detect(net);
    }, 10);
  };

  const detect = async (net) => {
    // Check data is available
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Set canvas height and width
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      // Make Detections
      const obj = await net.detect(video);

      // Draw mesh
      const ctx = canvasRef.current.getContext("2d");
      drawRect(obj, ctx); 
    }
  };

  useEffect(()=>{runCoco()},[]);

  return (
    <div className="App">
      <main className="App-main">
        <div className="App-container-cam">
          <Webcam
            ref={webcamRef}
            muted={true}
            className="App-webcam"
          />

          <canvas
            ref={canvasRef}
            className="App-webcam"
          />
        </div>
        <div className="App-container-text">
          <h1 class="App-title">Terminator</h1>
          <div>
            <p class="description">
              Identifica objectos contidianos<br />
              Este modelo fue entrenado con un set de datos de tensorflow
            </p>
            <p class="App-text-footer">Hecho por el CIPA 3</p>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
