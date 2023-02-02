/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package generaldpalgos;

import dptressexpm.twotreesEM.*;
import dptressexpm.twotreesEM.*;
import dptressexpm.twotreesEM.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.Path;

/**
 *
 * @author Matej Mihelcic, University of Eastern Finland implemented the dptressxpm.twotreesEM package as an extension of the 
 * original tree package decTreeWholeDP created for the manuscript: "Embedding differential privacy in decision tree
algorithm with different depths" by authors Xuanyu BAI , Jianguo YAO*, Mingxuan YUAN, Ke DENG,
 */
public class Settings {
    public String inputFile1 = "";
    public String inputFile2 = "";
    public String outputFile = "";
    public String targetSelectionFile = "";
    public int numIterations = 1;
    public int numInitTrials = 1;
    public int maxTreeDepth = 1;
    public int maxMCMCIteration = 1;
    public double equilibriumTreshold = 0.000000001;
    public double maxRedSupport = 0.8;
    public double minRedSupport = 0.05;
    public double minJaccard = 0.1;
    public double maxPval = 0.01;
    public double budget = 1.0;
    public int numClauses = 3;
    public double w1 = 0.5;
    public double w2 = 0.5;
    
    public Settings(){
         inputFile1 = "input1.arff";
         inputFile2 = "input2.arff";
         outputFile = "redescriptions.rr";
         targetSelectionFile = "targetSelectionFile.txt";
    }
    
    public Settings(File input){
        Path p = Paths.get(input.getAbsolutePath());
        
        try{
                BufferedReader read = Files.newBufferedReader(p);
                String line = "";
                
                while((line = read.readLine())!=null){
                    String tmp[] = line.split("=");
                    if(tmp[0].trim().contains("inputFile1")){
                        inputFile1 = tmp[1].trim();
                    }
                    else if(tmp[0].trim().contains("inputFile2")){
                        inputFile2 = tmp[1].trim();
                    }
                    else if(tmp[0].trim().contains("targetSelectionFile")){
                        targetSelectionFile = tmp[1].trim();
                    }
                    else if(tmp[0].trim().contains("numIterations")){
                        numIterations = Integer.parseInt(tmp[1].trim());
                    }
                    else if(tmp[0].trim().contains("numInitTrials")){
                        numInitTrials = Integer.parseInt(tmp[1].trim());
                    }
                    else if(tmp[0].trim().contains("maxTreeDepth")){
                        maxTreeDepth = Integer.parseInt(tmp[1].trim());
                    }
                     else if(tmp[0].trim().contains("maxMCMCIteration")){
                        maxMCMCIteration = Integer.parseInt(tmp[1].trim());
                    }
                     else if(tmp[0].trim().contains("equilibriumTreshold")){
                        equilibriumTreshold = Double.parseDouble(tmp[1].trim());
                    }
                     else if(tmp[0].trim().contains("maxRedSupport")){
                        maxRedSupport = Double.parseDouble(tmp[1].trim());
                    }
                     else if(tmp[0].trim().contains("minRedSupport")){
                        minRedSupport = Double.parseDouble(tmp[1].trim());
                    }
                     else if(tmp[0].trim().contains("minJaccard")){
                        minJaccard = Double.parseDouble(tmp[1].trim());
                    }
                     else if(tmp[0].trim().contains("maxPval")){
                        maxPval = Double.parseDouble(tmp[1].trim());
                    }
                     else if(tmp[0].trim().contains("totalPrivacyBudget")){
                        budget = Double.parseDouble(tmp[1].trim());
                    }
                    else if(tmp[0].trim().contains("maxNumClauses")){
                        numClauses = Integer.parseInt(tmp[1].trim());
                    }
                    else if(tmp[0].trim().contains("outputFile")){
                        outputFile = tmp[1].trim();
                    }
                     else if(tmp[0].trim().contains("weightSplit")){
                        w1 = Double.parseDouble(tmp[1].trim());
                    }
                     else if(tmp[0].trim().contains("weightNoise")){
                          w2 = Double.parseDouble(tmp[1].trim());
                     }
                }
                
        }
        catch(IOException e){
            e.printStackTrace();
        }
    }  
}
