/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package generaldpalgos;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

/**
 *
 * @author Matej
 */
public class ExtractMeasuresFromFilesRev2 {
    public static void main(String args[]){
        //String input = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\ResRev2\\Mammals\\Pooled\\redescriptionsMammalsRev2A0S1Pooled_named.queries";
        //String input = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\ResRev2\\B01Pooled\\redescriptionsCMS2_16Rev2B01A1S0Pooled.queries";
        // String input = "F:\\Matej Dokumenti\\Clanci - u tijeku\\DPRM\\Grafovi\\experiments\\DPAlgos\\CMS\\redescriptionsCMS2_16Emb.queries";
        //String input = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\OtherTBDPRMAlgos\\CMS2_16\\MadreBB01\\redescriptionsCMS2_16PooledMDBB01.queries";
       
        String names[] = {"NPAS", "Mammals", "MIMIC", "CMS2_1", "CMS2_4", "CMS2_8", "CMS2_16"};
        String base = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\ResRev2\\";
        String input = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\ResRev2\\";
        String input1 = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\ResRev2\\";
        
        String inR;
        String inR1;
        
    for(String ext:names){    
        inR= base+ext+"\\Pooled\\redescriptions"+ext+"Rev2";
        inR1= base + ext+"\\Pooled\\redescriptions"+ext+"Rev2";
        for(int alg = 0; alg<3;alg++){
            for(int st = 0;st<2;st++){
                if(alg<2){
                   input=inR+"A"+alg+"S"+st+"Pooled.queries";
                   input1=inR1+"A"+alg+"S"+st+"Pooled_named.queries";
                }
                else{
                   input=inR+"A"+alg+"B"+st+"Pooled.queries";
                   input1=inR1+"A"+alg+"B"+st+"Pooled_named.queries"; 
                }
        File in = new File(input);
        ArrayList<Double> js = new ArrayList<>();
        ArrayList<Double> pv = new ArrayList<>();
         ArrayList<Double> support = new ArrayList<>();
         int count = 0;
        try{
            Path p = Paths.get(in.getAbsolutePath());
            BufferedReader read = Files.newBufferedReader(p, StandardCharsets.UTF_8);
            String line = "";
            
            while((line = read.readLine())!=null){
                if(count == 0){
                    count = 1;
                    continue;
                }
                String tmp[] = line.split("\t");
                js.add(Double.parseDouble(tmp[3]));
                pv.add(Double.parseDouble(tmp[4]));
                support.add(Double.parseDouble(tmp[7]));
            }
            read.close();
            //FileWriter fw = new FileWriter("F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\ResRev2\\Mammals\\Pooled\\redescriptionsMammalsRev2A0S1PooledS.queries");
           // FileWriter fw = new FileWriter("F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\ResRev2\\B01Pooled\\redescriptionsCMS2_16Rev2B01A1S0PooledNS.queries");
            //FileWriter fw = new FileWriter("F:\\Matej Dokumenti\\Clanci - u tijeku\\DPRM\\Grafovi\\experiments\\DPAlgos\\CMS\\redescriptionsCMS2_16EmbNS.queries");
           // FileWriter fw = new FileWriter("F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\OtherTBDPRMAlgos\\CMS2_16\\MadreBB01\\redescriptionsCMS2_16PooledMDBB01JPSN.queries");
          
           String output = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\ResRev2\\";
           if(alg<2)
              output+=ext+"\\Pooled\\redescriptions"+ext+"Rev2A"+alg+"S"+st+"PooledNS.queries";
           else output+=ext+"\\Pooled\\redescriptions"+ext+"Rev2A"+alg+"B"+st+"PooledNS.queries";
           
          FileWriter fw = new FileWriter(output);
            
            for(int i=0;i<js.size();i++)
                fw.write(js.get(i)+" "+pv.get(i)+" "+support.get(i)+"\n");
            fw.close();
            
             in = new File(input1);
            js = new ArrayList<>();
            pv = new ArrayList<>();
           support = new ArrayList<>();
            count = 0;
          
             p = Paths.get(in.getAbsolutePath());
            read = Files.newBufferedReader(p, StandardCharsets.UTF_8);
            line = "";
            
            while((line = read.readLine())!=null){
                if(count == 0){
                    count = 1;
                    continue;
                }
                String tmp[] = line.split("\t");
                js.add(Double.parseDouble(tmp[3]));
                pv.add(Double.parseDouble(tmp[4]));
                support.add(Double.parseDouble(tmp[7]));
            }
            read.close();
            
            output = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\ResRev2\\";
           if(alg<2)
              output+=ext+"\\Pooled\\redescriptions"+ext+"Rev2A"+alg+"S"+st+"PooledS.queries";
           else output+=ext+"\\Pooled\\redescriptions"+ext+"Rev2A"+alg+"B"+st+"PooledS.queries";
           
            fw = new FileWriter(output);
            
            for(int i=0;i<js.size();i++)
                fw.write(js.get(i)+" "+pv.get(i)+" "+support.get(i)+"\n");
            fw.close();
            
        }
        catch(IOException e){
            e.printStackTrace();
        }
    }
        } }
    
    }
}
