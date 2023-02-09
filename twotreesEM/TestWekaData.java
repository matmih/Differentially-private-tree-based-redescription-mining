/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dptressexpm.twotreesEM;

import dptressexpm.twotreesEM.*;
import dptressexpm.twotreesEM.*;
import dptressexpm.twotreesEM.*;
import dptressexpm.twotreesEM.TwoTrees.Node;
import java.util.ArrayList;
import java.util.HashSet;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author Matej Mihelcic, University of Eastern Finland implemented the dptressxpm.twotreesEM package as an extension of the 
 * original tree package decTreeWholeDP created for the manuscript: "Embedding differential privacy in decision tree
algorithm with different depths" by authors Xuanyu BAI , Jianguo YAO*, Mingxuan YUAN, Ke DENG,
 */
public class TestWekaData {
    static public void main(String [] args){
        String inputFile = new String("test.arff");
        try{
             Instances data = (new ConverterUtils.DataSource(inputFile)).getDataSet();
            
             System.out.println("Original data" );
             System.out.println(data);
             
             Instances data1 = new Instances(data);
             
             System.out.println();
             System.out.println("Copied data");
             System.out.println(data1);
             
        ArrayList<String> values = new ArrayList<>(); /* FastVector is now deprecated. Users can use any java.util.List */
        values.add("c1"); values.add("c2"); values.add("c3"); values.add("c4");
        data1.setClassIndex(-1);
        data1.deleteAttributeAt(data1.numAttributes()-1);
        data1.insertAttributeAt(new Attribute("NewNominal", values), data1.numAttributes());
        
         System.out.println("Original data" );
             System.out.println(data);
             
             System.out.println();
             System.out.println("Copied data");
             System.out.println(data1);
             
             Instances c2 = new Instances(data1, data1.numInstances());
             c2.add(data1.get(2));
             c2.add(data.get(5));
        
             c2.setClassIndex(c2.numAttributes()-1);
             c2.get(0).setClassValue("c1");
             c2.get(1).setClassValue("c2");
             
             System.out.println();
             System.out.println("Copied data");
             System.out.println(data1);
             
             data1.setClassIndex(data1.numAttributes()-1);
             data1.get(0).setClassValue("c1");
             data1.get(3).setClassValue("c4");
             
              System.out.println();
             System.out.println("Copied data");
             System.out.println(data1);
             
             data.setClassIndex(data.numAttributes()-1);
             
             int classIndex = data.classIndex();
             Instances data2 = new Instances(data);

             data2.setClassIndex(-1);
             data2.deleteAttributeAt(classIndex);

             TwoTrees tt = new TwoTrees();
             tt.setMaxDepth(3);
             tt.setminNodeSize(0);
             tt.buildClassifierTest(data);
             
             System.out.println(tt.toString(0));
             
             
             int cl = tt.countLeafs(tt.getFirstRoot(), 0);
             
             System.out.println("Num leafs: "+cl);
             
              ArrayList<String> values1 = new ArrayList<>();
              
              for(int i=0;i<cl;i++)
                  values1.add("c"+i);
             
             data2.insertAttributeAt(new Attribute("NewNominal",values1), data2.numAttributes());
             
             
             HashSet<Integer> instInNode = new HashSet<>();
             
             for(int i=0;i<data2.numInstances();i++)
                 instInNode.add(i);
             
             data2 = tt.setTargets(tt.getFirstRoot(), classIndex, data2, instInNode);
             
             data2.setClassIndex(classIndex);
             
             System.out.println("Data with assigned class: ");
             System.out.println(data2);
             
             
             tt.updateTargetDistribution(tt.getFirstRoot(), data2);
             System.out.println(tt.toString(0));
             
             Node tmp = tt.getFirstRoot();
             
             while(!tmp.isLeaf)
                 if(tmp.children.length>1)
                        tmp = tmp.children[1];
                 else tmp = tmp.children[0];
             
             Redescription red = new Redescription();
             
             red.redNodes.add(new ArrayList<>());
             
             Node chp[] = tt.getFirstRoot().children;
             
             for(int i=0;i<chp.length;i++)
                 if(i%2 == 0){
                     Node tmp1 = chp[i];
                     while(!tmp1.isLeaf)
                         tmp1 = tmp1.children[0];
                     red.redNodes.get(0).add(tmp1);//get the leaf node of this node
                 }
             
             System.out.println();
             System.out.println("Elementary conjunction: ");
             String elConj=red.createElConjunctionString(tmp);
             System.out.println(elConj);
             
             System.out.println("Entities in a node: ");
            HashSet<Integer> instInNode1 = new HashSet<>();
             
             for(int i=0;i<data2.numInstances();i++)
                 instInNode1.add(i);
             
             HashSet<Integer> entities = tt.computeLeafEntityIDs(tmp, data, instInNode1);
             for(int i:entities)
                 System.out.print(i+" ");
             
             System.out.println();
             
             System.out.println();
             System.out.println("Query string: ");
             String qs = red.createQueryString(0);
             System.out.println(qs);
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }
}
