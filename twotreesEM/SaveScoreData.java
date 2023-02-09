/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dptressexpm.twotreesEM;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author mmihelci
 */
public class SaveScoreData {

  public static void main(String [] args){
        
         if(args.length==0){
            System.err.println("A path to the settings file must be provided!");
            System.err.println("Terminating execution!");
            System.exit(-1);
        }
         
         ArrayList<Double> treescores = new ArrayList<>();
         HashMap<Double, ArrayList<Double>> treeScoresBudget = new HashMap<>();
        
        File settingsPath = new File(args[0].trim());
        
        Settings set = new Settings(settingsPath);
        
        String inputFile1 = set.inputFile1;//new String("C:\\Users\\mmihelci\\Documents\\NetBeansProjects\\DPTressExpM\\BioW1.arff");
        String inputFile2 = set.inputFile2;//new String("C:\\Users\\mmihelci\\Documents\\NetBeansProjects\\DPTressExpM\\BioW2.arff");
        Instances dataW1 = null, dataW2 = null;    
        
        try{
                dataW1 = (new ConverterUtils.DataSource(inputFile1)).getDataSet();
                dataW2 = (new ConverterUtils.DataSource(inputFile2)).getDataSet();
                
                ArrayList<Redescription> AllRedescriptions = new ArrayList<>();
                
                int numIterations = set.numIterations;//80;
                
                 int numAttrs = dataW1.numAttributes();
                 int numAttrs1 = dataW2.numAttributes();
                 
                 numIterations = Math.min(numIterations, numAttrs+numAttrs1);
                
                
                double epsilon = set.budget/(double)numIterations;
                System.out.println("Epsilon: "+epsilon);

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
                  dataW1 = (new ConverterUtils.DataSource(inputFile1)).getDataSet();
                dataW2 = (new ConverterUtils.DataSource(inputFile2)).getDataSet();
                 //generate a random number deciding view from which to take an attribute
                 double probSim = rand.nextDouble();
                 
                int rind = 0;
               
                        if(probSim<=prob)
                            rind = rand.nextInt(dataW1.numAttributes());
                        else  rind = rand.nextInt(dataW2.numAttributes())+dataW1.numAttributes();
               
                
             for(double d=0.0001;d<100.0;d = d*10){   
                  dataW1 = (new ConverterUtils.DataSource(inputFile1)).getDataSet();
                dataW2 = (new ConverterUtils.DataSource(inputFile2)).getDataSet();
                 TwoTrees tt = new TwoTrees();
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
                epsilon = d;
                tt.setEpsilon(epsilon+"");
                tt.setMaxIteration(set.maxMCMCIteration);
                tt.setEquilibriumThreshold(set.equilibriumTreshold);
               // Attribute r = dataW1.attribute(20);
               
               if(numIterations>=0.9*(numAttrs+numAttrs1))
                   rind = it;
               
               while(usedIndex.contains(rind) && numIterations<0.9*(numAttrs+numAttrs1)){
                   if(probSim<=prob)
                       rind =  rand.nextInt(dataW1.numAttributes()); 
                   else
                        rind =  rand.nextInt(dataW1.numAttributes()+dataW2.numAttributes());   
               }
               
              // usedIndex.add(rind);
               System.out.println("Attribute index: "+rind);
               
                Attribute r = null;
                        
                        if(rind<dataW1.numAttributes() && probSim<=prob)
                                r = dataW1.attribute(rind);
                        else if(rind>=dataW1.numAttributes() && probSim>prob)
                                r = dataW2.attribute(rind-dataW1.numAttributes());
                        else if(probSim<=prob){
                            int chosen = 0;
                            for(int zz = 0; zz<dataW1.numAttributes();zz++)
                                if(!usedIndex.contains(zz)){
                                    rind = zz;
                                   // usedIndex.add(rind);
                                    chosen = 1;
                                    r = dataW1.attribute(rind);
                                    break;
                                } 
                            if(chosen == 0)
                                r = dataW1.attribute(rand.nextInt(dataW1.numAttributes()));
                        }
                        else if(probSim>prob){
                            int chosen = 0; 
                            for(int zz = 0; zz<dataW2.numAttributes();zz++)
                                if(!usedIndex.contains(zz)){
                                    rind = dataW1.numAttributes()+zz;
                                   // usedIndex.add(rind);
                                    chosen = 1;
                                     r = dataW2.attribute(rind-dataW1.numAttributes());
                                    break;
                                }  
                            if(chosen == 0)
                                 r = dataW2.attribute(rand.nextInt(dataW2.numAttributes()));
                        }
                int numValues = r.numValues();//number of classess
                
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
                        else s--;
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
                    }
                    
                System.out.println("CA: "+data1.classAttribute().toString()); 
                 if(probSim<=prob){
                        treescores = tt.createTreesScores(data1, dataW2);
                 }
                 else{
                     treescores = tt.createTreesScores(data1, dataW1);
                 }
                 
                treeScoresBudget.put(epsilon, treescores);
                 
                tt.setNumClausesRed(set.numClauses);
                tt.setMaxSupport(maxSupport);
                tt.setMinSupport(minSupport);
                tt.setMinJaccard(set.minJaccard);
                tt.setMaxPval(set.maxPval);
                System.out.println();
                System.out.println("Redescription creation started...");   
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
             
             }
             
             Collections.sort(AllRedescriptions, Collections.reverseOrder());
             
             System.out.println();
             System.out.println();
             System.out.println("Totali produced: "+AllRedescriptions.size()+" redescriptions");
             System.out.println("All produced redescriptions: ");
             
             //write scores only
            
             Iterator<Double> it = treeScoresBudget.keySet().iterator();
             double eps = 0.0;
             
             while(it.hasNext()){
             eps = it.next();
             FileWriter fw = new FileWriter("treeScores"+eps+".txt");
             treescores = treeScoresBudget.get(eps);
             
             for(int i=0;i<treescores.size();i++)
                 fw.write(treescores.get(i)+" ");
             System.out.println("Written.... "+eps);
             fw.close();
             }
             /*String rereMiOut = set.outputFile.split(".rr")[0].trim()+".queries";
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
                
                 String rq1 = AllRedescriptions.get(i).queryStringsReReMi.get(0);
                 String rq2 = AllRedescriptions.get(i).queryStringsReReMi.get(1);
                 
                 fw1.write("r"+(++rCount)+"\t"+rq1+"\t"+rq2+"\t"+AllRedescriptions.get(i).Jaccard+"\t"+AllRedescriptions.get(i).pval+"\t"+AllRedescriptions.get(i).querySupports.get(0)+"\t"+AllRedescriptions.get(i).querySupports.get(1)+"\t"+AllRedescriptions.get(i).support+"\t"+(dataW1.numInstances()-AllRedescriptions.get(i).supportUnion)+"\n");                
             }
             
             //implement a redescription evaluation function from data.
             //implement numerical attributes
             fw.close();
             fw1.close();*/
             
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }
}