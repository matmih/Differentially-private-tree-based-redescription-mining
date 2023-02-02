/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package generaldpalgos;

import java.io.Serializable;
import java.util.Enumeration;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * This class is developed by the authors of the manuscript: "Embedding differential privacy in decision tree
algorithm with different depths" by authors Xuanyu BAI , Jianguo YAO*, Mingxuan YUAN, Ke DENG, Xike Xie and Haibing Guan.
*It contains measures for differentially private algorithms DecTreeWholeDP and DecTreeExpMech.
 */
public class Node implements Serializable{
		
		private static final long serialVersionUID = 1L;     

		public Instances data; 
		
		public int depth,complement = 0;
		public int li;
		public Attribute splitAttr;
                public double splitValue;
		
		public Node[] children;
		
		public Node parent;
		
		public int index;
		
		public boolean isLeaf;
		
		public double[] count;             
		public double[] dist;              
		
		public void updateDist(){                      
			dist = count.clone();                        
			if( Utils.sum(dist) != 0.0)                      
				Utils.normalize(dist);      
		}
                
                 public Node(){
                    this.depth = 0;                         
			this.isLeaf = true;
			this.count = null;
			this.dist = null;
			this.splitAttr = null;
                        this.parent = null;
                        this.complement = 0;
                }
                 
                  public void copyNodeFull(Node another){
                    this.data = new Instances(another.data); 
			this.depth = another.depth;                         
			this.isLeaf = another.isLeaf;
			this.count = another.count;
			this.dist = another.dist;
			this.splitAttr = another.splitAttr;
                        this.parent = another.parent;
                        this.index = another.index;
                }
		
		public Node(Instances data, Node parent, int index){
			
			this.data = data;
			this.parent = parent;
			this.index = index;

			if(parent!=null)
				this.depth = parent.depth + 1;
			else
				this.depth = 1;                  
			
			double[] tempcount = new double[data.numClasses()];                   
			Enumeration<Instance> instEnum = data.enumerateInstances();  
			while(instEnum.hasMoreElements()){                              
				Instance inst = (Instance)instEnum.nextElement();            
				tempcount[(int)inst.classValue()]++;                  
			}
				
			this.count = tempcount;                                   
			this.dist = count.clone();                     
			if(Utils.sum(this.dist) != 0.0)                       
				Utils.normalize(this.dist);              
		}
		
        public Node(Node another){                              
			
			this.data = new Instances(another.data); 
			this.depth = another.depth;                         
			this.isLeaf = another.isLeaf;
			this.count = another.count;
			this.dist = another.dist;
			this.splitAttr = another.splitAttr;
        }
	}
        
