package generaldpalgos;

import java.io.Serializable;

/**
 *
 * This class is developed by the authors of the manuscript: "Embedding differential privacy in decision tree
algorithm with different depths" by authors Xuanyu BAI , Jianguo YAO*, Mingxuan YUAN, Ke DENG, Xike Xie and Haibing Guan.
*It contains measures for differentially private algorithms DecTreeWholeDP and DecTreeExpMech.
 */

public class GiniIndexDirectAdd implements Serializable{
	
	private static final long serialVersionUID = 1L;     
        
        public double score(Node node, double totalNumInstances){
		
		double finalscore = 0;
	
		if(node==null)
			return 0;
		if(node.data.numInstances() <= 0)                  
			return 0;
		
		if(node.isLeaf == true)                     
		{
			finalscore = leafScore(node);
			return finalscore;
		}
		else{
			for(Node child : node.children)               
			{
				double childscore = score(child);					
				if(child.isLeaf==true){
					finalscore -= ((double)child.data.numInstances()/totalNumInstances) * childscore; 
				}
				else{
					finalscore += childscore;
				}
			}
			
			return finalscore;                                     
		}	
	}


	public double score(Node node){
		
		double finalscore = 0;
	
		if(node==null)
			return 0;
		if(node.data.numInstances() <= 0)                  
			return 0;
		
		if(node.isLeaf == true)                     
		{
			finalscore = leafScore(node);
			return finalscore;
		}
		else{
			for(Node child : node.children)               
			{
				double childscore = score(child);					
				if(child.isLeaf==true){
					finalscore -= ((double)child.data.numInstances()) * childscore; 
				}
				else{
					finalscore += childscore;
				}
			}
			
			return finalscore;                                     
		}	
	}

	private double leafScore(Node node){                    
		
		double leafscore = 1;
		int numInst = node.data.numInstances();       
			
		if(numInst == 0){                           
			return 0;
		}
			
		double leafcount[] = node.count;            
		for(int i=0; i< leafcount.length; ++i){      
			if( leafcount[i] > 0){
				double score = Math.pow(leafcount[i]/numInst, 2);
				leafscore -= score;                            
			}
		}
			
		return leafscore;                                     
	}
	
}
