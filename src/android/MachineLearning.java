package coldova_plugin_ml;

import org.apache.cordova.CordovaPlugin;

import com.sun.glass.ui.MenuItem.Callback;

import org.apache.cordova.CallbackContext;

import java.lang.*;
import java.io.*;
import java.util.ArrayList;
import java.awt.image.BufferedImage;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import javax.imageio.ImageIO;

import org.json.*;
import org.nd4j.linalg.api.ndarray.*;
import org.nd4j.linalg.factory.Nd4j;

import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.neochi.ml.*;

/**
 * This class echoes a string called from JavaScript.
 */
public class MachineLearning extends CordovaPlugin {

    @Override
    public boolean execute(String action, JSONArray args, CallbackContext callbackContext) throws JSONException {
        if (action.equals("coolMethod")) {
            String message = args.getString(0);
            this.coolMethod(message, callbackContext);
            return true;
        }else if(action.equals("learn")){
            String json = args.getString(0);
            this.learn(json, callbackContext);
            return true;
        }
        return false;
    }

    private void coolMethod(String message, CallbackContext callbackContext) {
        if (message != null && message.length() > 0) {
            callbackContext.success(message);
        } else {
            callbackContext.error("Expected one non-empty string argument.");
        }
    }

    private void learn(String json, CallbackContext callbackContext){

        JSONObject JObject = new JSONObject(jsonData);
        JSONArray dataSets = JObject.getJSONArray("dataSets");

        JSONArray dataArray = dataSets.getJSONObject(0).getJSONArray("dataArray");

        ArrayList<String> fileList = new ArrayList<String>();
        ArrayList<Double> target = new ArrayList<Double>();
        String dir = null;
        try{
            for(int n=0; n<dataArray.length(); n++) {
                JSONObject data = dataArray.getJSONObject(n);
    
                String imageFilePath = data.getString("imageFilePath");
                String tag = data.getString("tag");
    
                if(dir == null){
                    new File(imageFilePath).getParent();
                }
    
                fileList.add(imageFilePath);
                if (tag.equals("sleeping")){
                    target.add(1.0);
                }else{
                    target.add(0.0);
                }
    
                System.out.println(imageFilePath);
                System.out.println(tag);
            }
        }catch(Exception e){
            callbackContext.error(e.getMessage());
            return;
        }
        
        try{
            // 画像のパスのリストから画像の配列（byte列）を取得
            INDArray indArray = null;//Nd4j.create();
            for(String path:fileList){
                INDArray img_vector = readImg2Vec(path);
                if(indArray == null){
                    indArray = img_vector.dup();
                }else{
                    indArray = Nd4j.vstack(indArray,img_vector);
                }
            }
        }catch(Exception e){
            callbackContext.error(e.getMessage());
            return;
        }

        double[] targetArr = new double[target.size()];
        for(int i=0; i < target.size(); i++){
            targetArr[i] = target.get(i);
        }
        INDArray y = Nd4j.create(targetArr, new int[]{target.size(), 1});

        try{
            // 学習実行
            NeochiClassifier cls = new NeochiClassifier((int)indArray.size(1));
            cls.fit(indArray, y, 10);
            //学習したモデルをJSONに吐き出す
            String modelJson = cls.toJson();
            String zipFilePath = writeJSONFile(modelJson, dir);
        }catch(Exception e){
            callbackContext.error(e.getMessage());
            return;
        }
        
        //作成したZIPファイルをアップロード
        try{
            uploadModel(zipFilePath);
        }catch(Exception e){
            callbackContext.error(e.getMessage());
        }
    }

    private static void uploadModel(String path) throws Excption{
        URL url = "";
        HttpURLConnection con = (HttpURLConnection) url.openConnection();

        con.setRequestMethod("POST");
        con.setDoOutput(true);

        byte[] b = Files.readAllBytes(Paths.get(path));
        OutputStream os = con.getOutputStream();
        os.write(b);

        os.close();
        con.disconnect();
    }

    private static String writeJSONFile(String json, String dir){
        try{
            String jsonFilePath = dir + "model.json";

            FileWriter fw = new FileWriter(jsonFilePath);
            fw.write(json);
            fw.close();

            String zipFilePath = dir + "model.zip";
            ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(zipFilePath));
            ZipEntry ze = new ZipEntry(jsonFilePath);
            zos.putNextEntry(ze);

            InputStream is = new FileInputStream(jsonFilePath);
            byte[] buf = new byte[1024];
            int len = 0;
            while((len = is.read(buf)) != -1){
                zos.write(buf, 0, len);
            }
            is.close();
            zos.closeEntry();
            zos.close();
            return zipFilePath;

        }catch (Exception e){
            e.printStackTrace();
            throw e;
        }
    }

    private static INDArray readImg2Vec(String path) throws Exception{
        File f = new File(path);
        try {
            BufferedImage img = ImageIO.read(f);

            int height, width;
            height = img.getHeight();
            width = img.getWidth();
            int color, r, g, b;

            double[] vector = new double[height*width];
            int i = 0;
            for(int y=0; y < height; y++){
                for(int x=0; x < width; x++){

                    color = img.getRGB( x, y );

                    r = ( color >> 16 ) & 0xff;
                    g = ( color >> 8 ) & 0xff;
                    b = color & 0xff;

                    // グレースケールに変換して０〜１に正規化
                    vector[i] = (r + g + b) / 3.0 / 255.0;
                    i++;
                }
            }

            return Nd4j.create(vector);
        }catch(Exception e){
            System.out.println(e);
            throw e;
        }
    }
}
