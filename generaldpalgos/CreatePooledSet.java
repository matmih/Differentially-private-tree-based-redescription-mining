/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package generaldpalgos;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 *
 * @author Matej
 */
public class CreatePooledSet {
    public static void main(String args[]){
        //String path = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\ResRev2\\";
       //String path = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\ResRev2\\CMSB01\\";
         String path = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\ResRev2\\OtherB01\\";
        String extension = "NPAS";
        String algo = "";
        int algoCode = 1, stabilized = 0;
        
        String input;
        /*String o = "redescriptions"+extension+"Rev2"+"A"+algoCode+"S"+stabilized+"Pooled.queries";
        if(algoCode == 2)
            o = "redescriptions"+extension+"Rev2"+"A"+algoCode+"B"+stabilized+"Pooled.queries";*/
        
         String o = "redescriptions"+extension+"Rev2B01"+"A"+algoCode+"S"+stabilized+"Pooled.queries";
        if(algoCode == 2)
            o = "redescriptions"+extension+"Rev2B01"+"A"+algoCode+"B"+stabilized+"Pooled.queries";
        
        //String output = path+"\\"+extension+"\\"+algo+"\\"+o;
        String output = path+"\\"+o;
        
        try{
            FileWriter fw = new FileWriter(output);
            BufferedReader br;
            Path p;
            
            /*for(int i=1;i<=100;i++){
                if(algoCode<2)
                    input = path+"\\"+extension+"\\"+algo+"\\"+"redescriptions"+extension+"Rev2R"+i+"A"+algoCode+"S"+stabilized+".queries";
                else input = path+"\\"+extension+"\\"+algo+"\\"+"redescriptions"+extension+"Rev2R"+i+"A"+algoCode+"B"+stabilized+".queries";
                p = Paths.get(input);
                br = Files.newBufferedReader(p, StandardCharsets.UTF_8);
                
                String line = "";
                
                int cnt = 0;
                while((line = br.readLine())!=null){
                    if(i==1 && cnt == 0){
                        fw.write(line+"\n");
                        cnt = 1;
                    }
                    else if(i!=1 && cnt == 0){ cnt = 1; continue;}
                    else fw.write(line+"\n");
                }
                br.close();
              }*/
            
            for(int i=1;i<=10;i++){
                if(algoCode<2)
                    input = path+"\\"+"redescriptions"+extension+"Rev2B01R"+i+"A"+algoCode+"S"+stabilized+".queries";
                else input = path+"\\"+"redescriptions"+extension+"Rev2B01R"+i+"A"+algoCode+"B"+stabilized+".queries";
                p = Paths.get(input);
                br = Files.newBufferedReader(p, StandardCharsets.UTF_8);
                
                String line = "";
                
                int cnt = 0;
                while((line = br.readLine())!=null){
                    if(i==1 && cnt == 0){
                        fw.write(line+"\n");
                        cnt = 1;
                    }
                    else if(i!=1 && cnt == 0){ cnt = 1; continue;}
                    else fw.write(line+"\n");
                }
                br.close();
              }
            
            fw.close();
        }
        catch(IOException e){
            e.printStackTrace();
        }
    }
}
