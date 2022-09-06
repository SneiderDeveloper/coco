import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const photoRef = useRef(null);
  const othercanvas = useRef(null);
  const resultado = useRef(null);

  const [modelo, setModelo] = useState(null)

  const getVideo = () => {
    navigator.mediaDevices.getUserMedia({ 
      video: { width: 500, height: 500 } 
    })
    .then(stream => {
      let video = videoRef.current
      video.srcObject = stream
      video.play()
      // procesarCamara();
      // predecir();
    })
    .catch(err => {
      console.log(err)
    })
  }

  const loadingModel = async () => {
    console.log("Cargando modelo...");
    const modelo = await tf.loadLayersModel('./model/model.json');
    setModelo(modelo)
    console.log("Modelo cargado...");
  }

  useEffect(() => {
    getVideo()
    const startModel = async () => {
      await loadingModel()
    }
    startModel()
  }, [videoRef])


var size = 400;
var camaras = [];

var currentStream = null;
var facingMode = "user"; //Para que funcione con el celular (user/environment)

function cambiarCamara() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => {
            track.stop();
        });
    }

    facingMode = facingMode == "user" ? "environment" : "user";

    var opciones = {
        audio: false,
        video: {
            facingMode: facingMode, width: size, height: size
        }
    };


    navigator.mediaDevices.getUserMedia(opciones)
        .then(function(stream) {
            currentStream = stream;
            videoRef.srcObject = currentStream;
        })
        .catch(function(err) {
            console.log("Oops, hubo un error", err);
        })
}

function predecir() {
    if (modelo != null) {
        //Pasar canvas a version 150x150
        resample_single(photoRef, 150, 150, othercanvas);

        var ctx2 = othercanvas.current.getContext("2d");

        var imgData = ctx2.getImageData(0,0,150,150);
        var arr = []; //El arreglo completo
        var arr150 = []; //Al llegar a arr150 posiciones se pone en 'arr' como un nuevo indice
        for (var p=0, i=0; p < imgData.data.length; p+=4) {
            var red = imgData.data[p]/255;
            var green = imgData.data[p+1]/255;
            var blue = imgData.data[p+2]/255;
            arr150.push([red, green, blue]); //Agregar al arr150 y normalizar a 0-1. Aparte queda dentro de un arreglo en el indice 0... again
            if (arr150.length == 150) {
                arr.push(arr150);
                arr150 = [];
            }
        }

        arr = [arr]; //Meter el arreglo en otro arreglo por que si no tio tensorflow se enoja >:(
        // Nah basicamente Debe estar en un arreglo nuevo en el indice 0, por ser un tensor4d en forma 1, 150, 150, 1
        var tensor4 = tf.tensor4d(arr);
        var resultados = modelo.predict(tensor4).dataSync();
        var mayorIndice = resultados.indexOf(Math.max.apply(null, resultados));

        var clases = ['Gato', 'Perro'];
        console.log("Prediccion", clases[mayorIndice]);
        document.getElementById("resultado").innerHTML = clases[mayorIndice];
    }

    setTimeout(predecir, 150);
}

function procesarCamara() {
    
    var ctx = photoRef.getContext("2d");

    ctx.drawImage(videoRef, 0, 0, size, size, 0, 0, size, size);

    setTimeout(procesarCamara, 20);
}

/**
    * Hermite resize - fast image resize/resample using Hermite filter. 1 cpu version!
    * 
    * @param {HtmlElement} canvas
    * @param {int} width
    * @param {int} height
    * @param {boolean} resize_canvas if true, canvas will be resized. Optional.
    * Cambiado por RT, resize canvas ahora es donde se pone el chiqitillllllo
    */
function resample_single(canvas, width, height, resize_canvas) {
    var width_source = canvas.width;
    var height_source = canvas.height;
    width = Math.round(width);
    height = Math.round(height);

    var ratio_w = width_source / width;
    var ratio_h = height_source / height;
    var ratio_w_half = Math.ceil(ratio_w / 2);
    var ratio_h_half = Math.ceil(ratio_h / 2);

    var ctx = canvas.current.getContext("2d");
    var ctx2 = resize_canvas.current.getContext("2d");
    var img = ctx.getImageData(0, 0, width_source, height_source);
    var img2 = ctx2.createImageData(width, height);
    var data = img.data;
    var data2 = img2.data;

    for (var j = 0; j < height; j++) {
        for (var i = 0; i < width; i++) {
            var x2 = (i + j * width) * 4;
            var weight = 0;
            var weights = 0;
            var weights_alpha = 0;
            var gx_r = 0;
            var gx_g = 0;
            var gx_b = 0;
            var gx_a = 0;
            var center_y = (j + 0.5) * ratio_h;
            var yy_start = Math.floor(j * ratio_h);
            var yy_stop = Math.ceil((j + 1) * ratio_h);
            for (var yy = yy_start; yy < yy_stop; yy++) {
                var dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half;
                var center_x = (i + 0.5) * ratio_w;
                var w0 = dy * dy; //pre-calc part of w
                var xx_start = Math.floor(i * ratio_w);
                var xx_stop = Math.ceil((i + 1) * ratio_w);
                for (var xx = xx_start; xx < xx_stop; xx++) {
                    var dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half;
                    var w = Math.sqrt(w0 + dx * dx);
                    if (w >= 1) {
                        //pixel too far
                        continue;
                    }
                    //hermite filter
                    weight = 2 * w * w * w - 3 * w * w + 1;
                    var pos_x = 4 * (xx + yy * width_source);
                    //alpha
                    gx_a += weight * data[pos_x + 3];
                    weights_alpha += weight;
                    //colors
                    if (data[pos_x + 3] < 255)
                        weight = weight * data[pos_x + 3] / 250;
                    gx_r += weight * data[pos_x];
                    gx_g += weight * data[pos_x + 1];
                    gx_b += weight * data[pos_x + 2];
                    weights += weight;
                }
            }
            data2[x2] = gx_r / weights;
            data2[x2 + 1] = gx_g / weights;
            data2[x2 + 2] = gx_b / weights;
            data2[x2 + 3] = gx_a / weights_alpha;
        }
    }


    ctx2.putImageData(img2, 0, 0);
}

  return (
    <div className="App">
      <main>
        <div>
          <video ref={videoRef}></video>
          <canvas ref={photoRef}></canvas>
          {/* <canvas ref={othercanvas} width="150" height="150" style="display: none"></canvas> */}
          <h4 ref={resultado}></h4>
          <button></button>
        </div>
        <div></div>
      </main>
    </div>
  );
}

export default App;
