/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dptressexpm.twotreesEM;

import dptressexpm.twotreesEM.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author Matej Mihelcic, University of Eastern Finland implemented the dptressxpm.twotreesEM package as an extension of the 
 * original tree package decTreeWholeDP created for the manuscript: "Embedding differential privacy in decision tree
algorithm with different depths" by authors Xuanyu BAI , Jianguo YAO*, Mingxuan YUAN, Ke DENG,
 */
public class TrackAndReproduce {
    public static void main(String [] args){
        
        int trackReproduce = 1; //1 - track, 2 - reproduce
        ArrayList<Integer> indexesInOrder = new ArrayList<>();
        
         if(args.length==0){
            System.err.println("A path to the settings file must be provided!");
            System.err.println("Terminating execution!");
            System.exit(-1);
        }
         
        File settingsPath = new File(args[0].trim());
        
        Settings set = new Settings(settingsPath);
        
        String inputFile1 = set.inputFile1;//new String("C:\\Users\\mmihelci\\Documents\\NetBeansProjects\\DPTressExpM\\BioW1.arff");
        String inputFile2 = set.inputFile2;//new String("C:\\Users\\mmihelci\\Documents\\NetBeansProjects\\DPTressExpM\\BioW2.arff");
        Instances dataW1 = null, dataW2 = null;  
        
        if(args.length == 2){
            trackReproduce = Integer.parseInt(args[1].trim());
            if(trackReproduce!=1 && trackReproduce!=2)
                trackReproduce = 1;
         }
        
        try{
                dataW1 = (new ConverterUtils.DataSource(inputFile1)).getDataSet();
                dataW2 = (new ConverterUtils.DataSource(inputFile2)).getDataSet();
                
                if(trackReproduce == 2){
                    Path p = Paths.get(set.targetSelectionFile);
                    BufferedReader r = Files.newBufferedReader(p);
                    String line = "";
                    
                    while((line = r.readLine())!=null){
                        String tmp[]= line.split(" ");
                        
                        for(int zz = 0; zz<tmp.length;zz++)
                            indexesInOrder.add(Integer.parseInt(tmp[zz].trim()));
                    }   
                    r.close();
                    //load data from a file
                }
                
                ArrayList<Redescription> AllRedescriptions = new ArrayList<>();
                
                int numIterations = set.numIterations;//80;
                
                 int numAttrs = dataW1.numAttributes();
                 int numAttrs1 = dataW2.numAttributes();
                 
                 numIterations = Math.min(numIterations, numAttrs+numAttrs1);
                
                 if(trackReproduce == 2){
                     numIterations = indexesInOrder.size();
                 }
                
                double epsilon = set.budget/(double)numIterations;

                HashSet<Integer> usedIndex = new HashSet<>();
                
                
                //create initial targets
               
                Random rand = new Random();
               // Attribute r = dataW1.attribute(rand.nextInt(dataW1.numAttributes()));
               
               int minSupport = 0, maxSupport = 0;
               
               if(set.minRedSupport<=1.0)
                   minSupport = (int)(dataW1.numInstances()*set.minRedSupport);
               else minSupport = (int)set.minRedSupport;
               
               if(set.maxRedSupport<=1.0)
                   maxSupport = (int)(dataW1.numInstances()*set.maxRedSupport);
               else maxSupport = (int)set.maxRedSupport;
               
               if(minSupport>maxSupport){
                   System.err.println("minSupport>maxSupport, change settings file!");
                   System.exit(-2);
               }
               
               int totalAttrs = numAttrs+numAttrs1;
               double prob = ((double)numAttrs)/((double) totalAttrs);//probability of choosing attr from view1
               
               
               
             for(int it=0;it<numIterations;it++){
                 System.out.println("Iteration: "+it+"/ "+numIterations);
                 
                 //generate a random number deciding view from which to take an attribute
                 double probSim = rand.nextDouble();
                 int rind = 0;
                 Attribute r = null;
                 
                 TwoTrees tt = new TwoTrees();
               if(trackReproduce == 1){
                 if(probSim<=prob){
                     System.out.println("Choosing view1!");
                 tt.view1Data = dataW1;
                 tt.view2Data = dataW2;
                 }
                 else{
                      System.out.println("Choosing view2!");
                   tt.view1Data = dataW2;
                   tt.view2Data = dataW1;  
                 }
                 tt.setMinSupportReal(minSupport);
                tt.setMaxDepth(set.maxTreeDepth);
                tt.setminNodeSize(minSupport);
                tt.setEpsilon(epsilon+"");
                tt.setWeights(set.w1, set.w2);
                tt.setMaxIteration(set.maxMCMCIteration);
                tt.setEquilibriumThreshold(set.equilibriumTreshold);
               // Attribute r = dataW1.attribute(20);
             
               
                        if(probSim<=prob)
                            rind = rand.nextInt(dataW1.numAttributes());
                        else  rind = rand.nextInt(dataW2.numAttributes())+dataW1.numAttributes();
               
               if(numIterations>=0.9*(numAttrs+numAttrs1))
                   rind = it;
               
               while(usedIndex.contains(rind) && numIterations<0.9*(numAttrs+numAttrs1)){
                   if(probSim<=prob)
                       rind =  rand.nextInt(dataW1.numAttributes()); 
                   else
                        rind =  rand.nextInt(dataW1.numAttributes()+dataW2.numAttributes());   
               }
               
               usedIndex.add(rind);
               System.out.println("Attribute index: "+rind);
               
             
                        
                        if(rind<dataW1.numAttributes() && probSim<=prob)
                                r = dataW1.attribute(rind);
                        else if(rind>=dataW1.numAttributes() && probSim>prob)
                                r = dataW2.attribute(rind-dataW1.numAttributes());
                        else if(probSim<=prob){
                            int chosen = 0;
                            for(int zz = 0; zz<dataW1.numAttributes();zz++)
                                if(!usedIndex.contains(zz)){
                                    rind = zz;
                                    usedIndex.add(rind);
                                    chosen = 1;
                                    r = dataW1.attribute(rind);
                                    break;
                                } 
                            if(chosen == 0){
                                rind = rand.nextInt(dataW1.numAttributes());
                                r = dataW1.attribute(rind);
                            }
                        }
                        else if(probSim>prob){
                            int chosen = 0; 
                            for(int zz = 0; zz<dataW2.numAttributes();zz++)
                                if(!usedIndex.contains(zz)){
                                    rind = dataW1.numAttributes()+zz;
                                    usedIndex.add(rind);
                                    chosen = 1;
                                     r = dataW2.attribute(rind-dataW1.numAttributes());
                                    break;
                                }  
                            if(chosen == 0){
                                rind = rand.nextInt(dataW2.numAttributes());
                                 r = dataW2.attribute(rind);
                            }
                        }
               }
               else{
                   rind = indexesInOrder.get(it);
                   
                 if(rind<dataW1.numAttributes()){
                     System.out.println("Choosing view1!");
                 tt.view1Data = dataW1;
                 tt.view2Data = dataW2;
                 }
                 else{
                      System.out.println("Choosing view2!");
                   tt.view1Data = dataW2;
                   tt.view2Data = dataW1;  
                 }
                tt.setMinSupportReal(minSupport);
                tt.setMaxDepth(set.maxTreeDepth);
                tt.setminNodeSize(minSupport);
                tt.setEpsilon(epsilon+"");
                tt.setMaxIteration(set.maxMCMCIteration);
                tt.setEquilibriumThreshold(set.equilibriumTreshold);
                   
                   
                   
                   if(rind<dataW1.numAttributes()){
                       r = dataW1.attribute(rind);
                       prob = 1.0; probSim = 0;
                   }
                   else{
                        r = dataW2.attribute(rind-dataW1.numAttributes());
                       prob = 0.0; probSim = 1.0;
                   }
               }
                      
               if(trackReproduce == 1)
                  indexesInOrder.add(rind);
               
                int numValues = r.numValues();//number of classess
                 if(trackReproduce == 2){
                     System.out.println("Run sequence loaded: ");
                     System.out.println("Attribute: "+rind);
                 }
                
                Instances data1 = null;
                
                if(probSim<=prob){
                        data1 = new Instances(dataW1);
                }
                else data1 = new Instances(dataW2);

                 ArrayList<String> values = new ArrayList<>(); /* FastVector is now deprecated. Users can use any java.util.List */
       
                if(!r.isNumeric()){ 
                  for(int i=0;i<numValues;i++)
                        values.add("c"+i);
                  data1.insertAttributeAt(new Attribute("NewNominal", values), data1.numAttributes());
                  data1.setClassIndex(data1.numAttributes()-1);
                  
                   for(int i=0;i<data1.numInstances();i++){
                    //  data1.get(i).
                    int in = 0;
                    String val = data1.get(i).stringValue(r.index());
                    for(int zz = 0; zz<r.numValues();zz++)
                        if(val.trim().equals(r.value(zz).trim())){
                            in = zz;
                            break;
                        }
                      data1.get(i).setClassValue(values.get(in));//check! (index out of bounds), seems to be fine
                  }
                  
                }
                else{
                    
                    HashSet<Double> attrV = new HashSet<>();
                    ArrayList<Double> attrValues = null;//new ArrayList<>();
                    
                    double perc = 0.92;
                    
                    for(int s = 0; s<perc*data1.numInstances();s++){
                        int ind = rand.nextInt(data1.numInstances());
                        if(perc>0.9)
                            ind = s;
                        if(!data1.get(ind).isMissing(r))
                             attrV.add(data1.get(ind).value(r));
                        else if(perc<0.9) 
                            s--;
                    }
                        //attrValues.add(data1.get(rand.nextInt(data1.numInstances())).value(r));
                      
                      attrValues = new ArrayList(attrV);
                      Collections.sort(attrValues);
                      double vp[] = new double[attrValues.size()];
                      
                      System.out.println("vp: ");
                      for(int s=0;s<attrValues.size();s++){
                          System.out.print(attrValues.get(s)+" ");
                          vp[s] = attrValues.get(s);
                      }
                      System.out.println();
                      
                      double n = Math.pow(data1.numInstances(), -0.33);
                      DescriptiveStatistics da = new DescriptiveStatistics(vp);
                      double iqr = da.getPercentile(75) - da.getPercentile(25);
                      double h = 2*iqr*n;
                      
                      System.out.println("h: "+h);
                      
                      if(h == 0)
                          continue;
                      
                      ArrayList<Double> splittingPoints = new ArrayList<>();
                      
                      double tmpInit = attrValues.get(0);
                      for(int s=1;s<attrValues.size();s++){
                              if(Math.abs(attrValues.get(s)-tmpInit)>h){
                                  splittingPoints.add(attrValues.get(s));
                                  tmpInit = attrValues.get(s);
                              }
                      }
                      
                     System.out.println("attrValues size: "+attrValues.size());
                     System.out.println("n: "+n);
                      
                       for(int s=0;s<splittingPoints.size()+2;s++){
                        values.add("c"+s);
                       }
                       System.out.println();
                  data1.insertAttributeAt(new Attribute("NewNominal", values), data1.numAttributes());
                  data1.setClassIndex(data1.numAttributes()-1);
                 // data1.setClass(data1.attribute(data1.numAttributes()-1));
                  
                  System.out.println("Splitting point size: "+splittingPoints.size());
                  int numAssigned = 0;
                  HashMap<Integer,Integer> classCounts = new HashMap<>();
                  
                    for(int s=0;s<data1.numInstances();s++){
                        
                        if(data1.get(s).value(r)<=splittingPoints.get(0)){
                             data1.get(s).setClassValue(values.get(0));
                             numAssigned++;
                             if(!classCounts.containsKey(0))
                                 classCounts.put(0, 1);
                             else classCounts.put(0,classCounts.get(0)+1);
                        }
                        else if(data1.get(s).value(r)>splittingPoints.get(splittingPoints.size()-1)){
                            data1.get(s).setClassValue(values.get(values.size()-1));
                            numAssigned++;
                            
                            if(!classCounts.containsKey(values.size()-1))
                                 classCounts.put(values.size()-1, 1);
                             else classCounts.put(values.size()-1,classCounts.get(values.size()-1)+1);
                            
                        }
                        else{
                            int a = 0;
                            for(int s1 = 1;s1<splittingPoints.size();s1++)
                                if(data1.get(s).value(r)<=splittingPoints.get(s1)){
                                      data1.get(s).setClassValue(values.get(s1));
                                         numAssigned++;
                                         a = 1;
                                         if(!classCounts.containsKey(s1))
                                                  classCounts.put(s1, 1);
                                          else classCounts.put(s1,classCounts.get(s1)+1);
                                         break;
                                }
                            if(a == 0){
                             data1.get(s).setClassValue(values.get(rand.nextInt(values.size())));
                             System.out.println("su: "+s);
                            }
                        }

                    }

                     System.out.println("Num assigned: "+numAssigned); 
                     
                     System.out.println("Class assignement: ");
                     for(int c:classCounts.keySet())
                         System.out.println(c+" "+classCounts.get(c));
                      
                   /* 
                     KMeansPlusPlusClusterer kmc = new KMeansPlusPlusClusterer(10);
                      ArrayList<DoublePoint> points = new ArrayList<>();
                      HashMap<DoublePoint, Integer> pm = new HashMap<>();
                      
                      for(int s=0;s<data1.numInstances();s++){
                          double tmpA[] = new double[1];
                          tmpA[0] = data1.get(s).value(r);
                          points.add(new DoublePoint(tmpA));
                          pm.put(points.get(s), s);
                      }
                      
                      List<Cluster<DoublePoint>> clusters = kmc.cluster(points);
                      
                      for(int s=0;s<clusters.size();s++)
                        values.add("c"+s);
                  data1.insertAttributeAt(new Attribute("NewNominal", values), data1.numAttributes());
                  data1.setClassIndex(data1.numAttributes()-1);
                  
                  HashSet<Integer> intCounter = new HashSet<>();
                  System.out.println("Num clusters: "+clusters.size());
                    for(int s=0;s<clusters.size();s++){
                        System.out.println("Cluster size: "+clusters.get(s).getPoints().size());
                        for(DoublePoint p : clusters.get(s).getPoints()) {
                            int index = pm.get(p);
                            data1.get(index).setClassValue(values.get(s));
                            intCounter.add(index);
                        }
                    }
                  System.out.println("Assigned entities: "+intCounter.size());
                  System.out.println("Num entities: "+pm.keySet().size());
                    
                    //cluster the entities to form classes
                  /*  ArrayList<Double> attrValues = new ArrayList<>();
                    ArrayList<Double> attrResiduals = new ArrayList<>();
                    
                    for(int s = 0; s<0.2*data1.numInstances();s++)
                        attrValues.add(data1.get(rand.nextInt(data1.numInstances())).value(r));
                        
                      Collections.sort(attrValues);
                      
                      for(int s = 0;s<attrValues.size();s++)
                          if(s>0 && s<attrValues.size()-1)
                             attrResiduals.add(Math.min(Math.abs(attrValues.get(s)-attrValues.get(s-1)),Math.abs(attrValues.get(s)-attrValues.get(s+1))));
                          else if(s == 0){
                             attrResiduals.add(Math.abs(attrValues.get(s)-attrValues.get(s+1)));
                          }
                          else if(s == attrValues.size()-1){
                              attrResiduals.add(Math.abs(attrValues.get(s)-attrValues.get(s-1)));
                          }
                      Collections.sort(attrResiduals);
                      
                      double maxDiff = Double.NEGATIVE_INFINITY;
                      int indexDiff = -1;
                      
                      for(int s = 0; s<attrResiduals.size();s++){
                           double tmpDiff = 0.0;
                          for(int s1 = 0;s1<5;s1++){
                              if(s1+s<attrResiduals.size())
                                    tmpDiff+=attrResiduals.get(s+s1);
                          }
                          if(tmpDiff>maxDiff){
                              maxDiff = tmpDiff;
                              indexDiff = s;
                          }
                      }
                      
                      double epsDB = attrResiduals.get(indexDiff);
                      System.out.println("epsDB: "+epsDB);
                      DBSCANClusterer db = new DBSCANClusterer(epsDB,(int)(0.1*data1.numInstances()));
                      ArrayList<DoublePoint> points = new ArrayList<>();
                      HashMap<DoublePoint, Integer> pm = new HashMap<>();
                      
                      for(int s=0;s<data1.numInstances();s++){
                          double tmpA[] = new double[1];
                          tmpA[0] = data1.get(s).value(r);
                          points.add(new DoublePoint(tmpA));
                          pm.put(points.get(s), s);
                      }
                      
                      List<Cluster<DoublePoint>> clusters = db.cluster(points);
                      
                      for(int s=0;s<clusters.size();s++)
                        values.add("c"+s);
                  data1.insertAttributeAt(new Attribute("NewNominal", values), data1.numAttributes());
                  data1.setClassIndex(data1.numAttributes()-1);
                  
                  HashSet<Integer> intCounter = new HashSet<>();
                  System.out.println("Num clusters: "+clusters.size());
                    for(int s=0;s<clusters.size();s++){
                        System.out.println("Cluster size: "+clusters.size());
                        for(DoublePoint p : clusters.get(s).getPoints()) {
                            int index = pm.get(p);
                            data1.get(index).setClassValue(values.get(s));
                            intCounter.add(index);
                        }
                    }
                  System.out.println("Assigned entities: "+intCounter.size());
                  System.out.println("Num entities: "+data1.numInstances());*/
                 /* for(int i=0;i<data1.numInstances();i++){
                    //  data1.get(i).
                      data1.get(i).setClassValue(values.get((int)data1.get(i).value(r.index())));
                  }*/
                      
                    }
                    
                System.out.println("CA: "+data1.classAttribute().toString()); 
                 if(probSim<=prob){
                        tt.createTrees(data1, dataW2);
                 }
                 else{
                     tt.createTrees(data1, dataW1);
                 }
//                 int numsimp = tt.leafMatches(data1, dataW2);
  //               System.out.println("Number of simple reds: "+numsimp);
                 
                tt.setMaxIterationRed(500);
                tt.setNumClausesRed(3);
                tt.setMaxSupport(maxSupport);
                tt.setMinSupport(minSupport);
                tt.setMinJaccard(set.minJaccard);
                tt.setMaxPval(set.maxPval);
                System.out.println();
                System.out.println("Redescription creation started...");
               // Redescription red = tt.createRedescription(dataW1, dataW2);
               //createRedescription(int numClauses, int maxIter, Instances data, Instances data1)
               
                
               /* System.out.println();
                System.out.println();
                System.out.println("Chosen redescription: ");
                System.out.println(red.queryStrings.get(0));
                System.out.println(red.queryStrings.get(1));
                System.out.println("JS: "+red.Jaccard);
                System.out.println("p-val: "+red.pval);
                System.out.println("Support: "+red.support);
                
                Redescription red1 = new Redescription(red);*/
                
                //red = tt.nosyStatistics(red, dataW1, dataW2);
                //red1 = tt.nosyStatistics1(red1, dataW1, dataW2);
                
               /* System.out.println();
                System.out.println();
                System.out.println("Chosen redescription noisy: ");
                System.out.println(red.queryStrings.get(0));
                System.out.println(red.queryStrings.get(1));
                System.out.println("JS: "+red.Jaccard);
                System.out.println("p-val: "+red.pval);
                System.out.println("Support: "+red.support);
                
                
                System.out.println();
                System.out.println();
                System.out.println("Chosen redescription noisy v2: ");
                System.out.println(red1.queryStrings.get(0));
                System.out.println(red1.queryStrings.get(1));
                System.out.println("JS: "+red1.Jaccard);
                System.out.println("p-val: "+red1.pval);
                System.out.println("Support: "+red1.support);*/
                
                
                System.out.println();
                System.out.println();
                System.out.println("Creating redescriptions directly using nosy counts: ");
                ArrayList<Redescription> tmp = null;
                 if(probSim<=prob){
                     int flip = 0;
                     tmp = tt.createRedescriptionsNoisy(dataW1, dataW2, set.numClauses, flip);
                 }
                 else{
                     int flip = 1;
                     tmp = tt.createRedescriptionsNoisy(dataW2, dataW1, set.numClauses, flip);
                 }
                AllRedescriptions.addAll(tmp);
                System.out.println("Num produced reds: "+tmp.size());
                for(int z=0;z<tmp.size();z++)
                    System.out.print(tmp.get(z).Jaccard+" ");
             }
             
             
             Collections.sort(AllRedescriptions, Collections.reverseOrder());
             
             System.out.println();
             System.out.println();
             System.out.println("Totali produced: "+AllRedescriptions.size()+" redescriptions");
             System.out.println("All produced redescriptions: ");
             
             FileWriter fw = new FileWriter(set.outputFile);
             String rereMiOut = set.outputFile.split(".rr")[0].trim()+".queries";
             FileWriter fw1 = new FileWriter(rereMiOut);
             int rCount = 0;
             
             for(int i=0;i<AllRedescriptions.size();i++){
                 System.out.println(AllRedescriptions.get(i).queryStrings.get(0));
                 System.out.println(AllRedescriptions.get(i).queryStrings.get(1));
                 System.out.println("JS: "+AllRedescriptions.get(i).Jaccard);
                 System.out.println("p-value: "+AllRedescriptions.get(i).pval);
                 System.out.println("query support sizes: "+AllRedescriptions.get(i).querySupports.get(0)+" "+AllRedescriptions.get(i).querySupports.get(1));
                 System.out.println("support size: "+AllRedescriptions.get(i).support);
                 System.out.println("query union size: "+AllRedescriptions.get(i).supportUnion);
                 System.out.println("\n\n");
                 
                  if(i == 0){
                     fw.write("Redescriptions: \n\n");
                     fw1.write("rid	query_LHS	query_RHS	acc	pval	card_Exo	card_Eox	card_Exx	card_Eoo	\n");
                  }
                     
                 fw.write("q1: "+AllRedescriptions.get(i).queryStrings.get(0)+"\n");
                 fw.write("q2: "+AllRedescriptions.get(i).queryStrings.get(1)+"\n");
                 fw.write("J: "+AllRedescriptions.get(i).Jaccard+"\n");
                 fw.write("p: "+AllRedescriptions.get(i).pval+"\n");
                 fw.write("Redescription support size: "+AllRedescriptions.get(i).support+"\n");
                 fw.write("Query support union size: "+AllRedescriptions.get(i).supportUnion+"\n");
                 fw.write("\n\n");
                
                 /* String tr[] = AllRedescriptions.get(i).queryStrings.get(0).split(" OR ");
                  
                  for(int z=0;z<tr.length;z++){
                       tr[z]=tr[z].replaceAll("<=", " <= ");
                      tr[z]=tr[z].replaceAll(">", " > ");
                      tr[z]=tr[z].replaceAll("([^<])=", "$1 = ");
                      if(tr[z].contains("NOT ")){
                          tr[z] = tr[z].replaceAll("\\(", "\\( ");
                          tr[z] = tr[z].replaceAll("\\)", " \\)");
                          continue;
                      }
                           
                      if(tr[z].contains(" AND "))
                           tr[z] = "( "+tr[z].trim()+" )";
                  }
                  
                 String rq1 = "";
                 
                 for(int z=0;z<tr.length;z++)
                     if((z+1)<tr.length)
                        rq1+=tr[z]+" OR ";
                     else rq1+=tr[z];
                    
                 rq1 = rq1.replaceAll("NOT ", "! ").replaceAll(" OR ", " | ").replaceAll(" AND ", " & ");
    
                 String rq2 = "";
                 
                 tr = AllRedescriptions.get(i).queryStrings.get(1).split(" OR ");
                  
                  for(int z=0;z<tr.length;z++){
                      tr[z]=tr[z].replaceAll("<=", " <= ");
                      tr[z]=tr[z].replaceAll(">", " > ");
                      tr[z]=tr[z].replaceAll("([^<])=", "$1 = ");
                      if(tr[z].contains("NOT "))
                          continue;
                      if(tr[z].contains(" AND "))
                           tr[z] = "( "+tr[z].trim()+" )";
                  }
                  
                  for(int z=0;z<tr.length;z++)
                     if((z+1)<tr.length)
                        rq2+=tr[z]+" OR ";
                     else rq2+=tr[z];
                  
                  rq2 = rq2.replaceAll("NOT ", "! ").replaceAll(" OR ", " | ").replaceAll(" AND ", " & ");
                 */
                 String rq1 = AllRedescriptions.get(i).queryStringsReReMi.get(0);
                 String rq2 = AllRedescriptions.get(i).queryStringsReReMi.get(1);
                 
                 fw1.write("r"+(++rCount)+"\t"+rq1+"\t"+rq2+"\t"+AllRedescriptions.get(i).Jaccard+"\t"+AllRedescriptions.get(i).pval+"\t"+AllRedescriptions.get(i).querySupports.get(0)+"\t"+AllRedescriptions.get(i).querySupports.get(1)+"\t"+AllRedescriptions.get(i).support+"\t"+(dataW1.numInstances()-AllRedescriptions.get(i).supportUnion)+"\n");                
             }
             
             //implement a redescription evaluation function from data.
             //implement numerical attributes
             fw.close();
             fw1.close();
             
              if(trackReproduce == 1){//write the tracking file
                  fw = new FileWriter(set.targetSelectionFile);
                  
                  for(int i=0;i<indexesInOrder.size();i++)
                      if((i+1)<indexesInOrder.size())
                        fw.write(indexesInOrder.get(i)+" ");
                      else  fw.write(indexesInOrder.get(i)+"");
                  
                  fw.close();
              }
             
             
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }
}

