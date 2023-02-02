package generaldpalgos;

import java.io.Serializable;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.apache.commons.math3.util.Pair;

import weka.classifiers.AbstractClassifier;
import weka.core.*;
//decision tree with DP embedding

/**
 *
 * This class is developed by the authors of the manuscript: "Embedding differential privacy in decision tree
algorithm with different depths" by authors Xuanyu BAI , Jianguo YAO*, Mingxuan YUAN, Ke DENG.
*It contains measures for differentially private algorithms DecTreeWholeDP and DecTreeExpMech.
*Matej Mihelcic implemented minor modifications to make the resulting redescription mining algorithm differentially private
 */

public class DecTreeWholeDp extends AbstractClassifier{

	private static final long serialVersionUID = 1L;              

    public static final MathContext MATH_CONTEXT = new MathContext(20, RoundingMode.DOWN);  
    
	private int MaxDepth;
	public void setMaxDepth(int maxDepth) {
		MaxDepth = maxDepth;
	}
	
	private BigDecimal Epsilon = new BigDecimal(1.0);      
    public void setEpsilon(String epsilon) {                         
	    if (epsilon!=null && epsilon.length()!=0)
	        Epsilon = new BigDecimal(epsilon,MATH_CONTEXT);
    }
    private BigDecimal splitbudget = new BigDecimal(1.0,MATH_CONTEXT);
    private BigDecimal splitmarkovbudget = new BigDecimal(1.0,MATH_CONTEXT);
    private BigDecimal noisebudget = new BigDecimal(1.0,MATH_CONTEXT);
    
    private Random Random = new Random(); 
    public void setSeed(int seed){                          
		Random = new Random(seed);
	}
    
    private int MaxIteration;                            
    public void setMaxIteration(int maxiteration){
    	MaxIteration = maxiteration;
    }
    
    private double EquilibriumThreshold;                 
    public void setEquilibriumThreshold(double equilibriumThreshold){
    	EquilibriumThreshold = equilibriumThreshold;
    }

    private double GiniSen = 2;       
    
	private Node Root;
	
	private Set<Pair<Attribute, Double>> Attributes = new HashSet<Pair<Attribute,Double>>();
	private Set<Node> InnerNodes = new HashSet<Node>();
        int numLeafs = 0;
	GiniIndexDirectAdd gini = new GiniIndexDirectAdd();
	
	
        void renumerateLeaves(Node n){
            if(n.parent==null){
                this.numLeafs = 0;  
            }
            
            if(n.isLeaf){
                n.li = this.numLeafs;
                this.numLeafs++;
                return;
            }
            
           for(int i=0;i<n.children.length;i++) 
            renumerateLeaves(n.children[i]);
            
        }
        
         public Instances copyTargets(Instances data1, Instances data2){
             Instances tmp = new Instances(data2);
             
             int classIndex = tmp.classIndex();
             
             if(classIndex>=0){
                 tmp.setClassIndex(-1);
                 tmp.deleteAttributeAt(classIndex);       
             }
               
             
             ArrayList<String> values = new ArrayList<>();
             
             for (int i=0;i<data1.classAttribute().numValues();i++) {
                values.add(data1.classAttribute().value(i));
             }
             
             int ind = tmp.numAttributes();
             tmp.insertAttributeAt(new Attribute("NewNominal",values), tmp.numAttributes());
             tmp.setClassIndex(ind);
             
             for(int i=0;i<data1.numInstances();i++)
                 if(!data1.get(i).classIsMissing())
                     tmp.get(i).setValue(ind, data1.get(i).stringValue(data1.classIndex()));
                 else tmp.get(i).setClassMissing();
                //  else 
             
             
             return tmp;
         }
         
          public Instances setTargets(Node node, int classIndex , Instances data, HashSet<Integer> instInNode){

            Attribute attr = node.splitAttr;
            
            if(node.isLeaf){
                //set targets and return
                for(int i:instInNode){
                    Instance inst = data.get(i);
                  //  System.out.println(inst.attribute(classIndex).name()+" "+classIndex);
                   // System.out.println("Node: "+"c"+node.li);
                    inst.setValue(classIndex, "c"+node.li);
                    data.set(i, inst);
                }
                
                return data;
            }
		
        if(!attr.isNumeric()){    
         for(int j=0;j<attr.numValues();j++){ 
             HashSet<Integer> instancesInANode = new HashSet<>();
            for(int i:instInNode)             
		{                     
			Instance inst = data.get(i); 
                        if(inst.isMissing(attr))
                            inst.setClassMissing();
                        else if(inst.stringValue(attr).trim().equals(attr.value(j).trim()))//prije preko double-a
                            instancesInANode.add(i);
		}
            data = setTargets(node.children[j], classIndex, data,  instancesInANode);
         }
       }
        else{
            for(int j=0;j<2;j++){
                HashSet<Integer> instancesInANode = new HashSet<>();
            for(int i:instInNode)             
		{
                     //handle missing values
			Instance inst = data.get(i); 
                        if(inst.isMissing(attr))//it is missing
                            inst.setClassMissing();                           
                        else if(j==0 && inst.value(attr)<=node.splitValue)
                            instancesInANode.add(i);
                        else if(j==1 && inst.value(attr)>node.splitValue)
                            instancesInANode.add(i);
		}
            data = setTargets(node.children[j], classIndex, data,  instancesInANode);
            }
        }
            
            return data;
        }
          
          
        public int countLeafs(Node node, int count){
            
            if(node.isLeaf){
                count = count+1;
                return count;
            }
            
            for(int i=0;i<node.children.length;i++)
                count = countLeafs(node.children[i],count);
            
            return count;
        }
          
         public Instances addTargets(Instances data){
            Instances tmp = new Instances(data);
            int classIndex = data.classIndex();
            
            tmp.setClassIndex(-1);
            tmp.deleteAttributeAt(classIndex);
            int cl = countLeafs(Root, 0);
        
            ArrayList<String> values1 = new ArrayList<>();
             // System.out.println("Num leaves: "+cl);
              for(int i=0;i<cl;i++)
                  values1.add("c"+i);
        
            tmp.insertAttributeAt(new Attribute("NewNominal",values1), tmp.numAttributes());
         
        HashSet<Integer> instInNode = new HashSet<>();
             
             for(int i=0;i<tmp.numInstances();i++)
                 instInNode.add(i);
             
             tmp.setClassIndex(tmp.numAttributes()-1);
             //System.out.println(toString(RootFirst));
             tmp = setTargets(Root, classIndex, tmp, instInNode);
             
            return tmp;
        } 
         
          //sample size, fraction of the input data
        public HashSet<Pair<Attribute, Double>> createAttrList(Instances data, double sampleSize){
            HashSet<Pair<Attribute, Double>> tmp = new HashSet<>();
            Random rand = new Random();
            
            if(sampleSize<0 || sampleSize>1)
                sampleSize = 0.2;
            
            int numInst = (int)(sampleSize * data.numInstances());
            HashSet<Integer> sampled = new HashSet<>();
            
            Instances tmpInstances = new Instances(data,0);
            
            
            for(int i=0;i<numInst;i++){
                
                if(sampled.size() == data.numInstances())
                    break;
                
                int index = rand.nextInt(data.numInstances());
                
                if(!sampled.contains(index)){
                    sampled.add(index);
                    tmpInstances.add(data.get(index));
                }
                else{
                    while(sampled.contains(index)){
                        index = rand.nextInt(data.numInstances());
                    }
                    
                     sampled.add(index);
                    tmpInstances.add(data.get(index));
                }
            }
            
            //create tmp from tmpInstances
            for(int i=0;i<tmpInstances.numAttributes();i++){
                HashSet<Double> values = new HashSet<>();
                
                 Attribute a = tmpInstances.attribute(i);
                 if(a == data.classAttribute())
                     continue;
                
                 if(a.isNumeric()){
                    for(int j=0;j<tmpInstances.numInstances();j++){
                        if(!tmpInstances.get(j).isMissing(a))
                            values.add(tmpInstances.get(j).value(i));
                    }

                for(double d:values){
                    tmp.add(new Pair<>(a,d));
                }
              }
                 else{
                     tmp.add(new Pair<>(a,Double.NEGATIVE_INFINITY));
                 }
                
            }
            
            return tmp;
        }
        
        public Node getRoot(){
            return Root;
        }

	public void buildClassifier(Instances data) throws Exception {
		 HashSet<Pair<Attribute,Double>> allAttributesGlobal = this.createAttrList(data, 0.8);
	HashSet<Pair<Attribute,Double>> allAttributes = new HashSet<>();
        Enumeration<Attribute> attrEnum = data.enumerateAttributes();  
        for(Pair<Attribute,Double> p:allAttributesGlobal)
                allAttributes.add(p);
        
        Attributes = new HashSet<Pair<Attribute,Double>>(allAttributes);

        Root = new Node(data, null, 0);                       
        
        splitbudget = Epsilon.divide(BigDecimal.valueOf(2),MATH_CONTEXT);
        noisebudget = splitbudget;
        splitmarkovbudget = Epsilon;//.subtract(noisebudget,MATH_CONTEXT);
         System.out.println("Budget markov: "+splitmarkovbudget);
        Set<Pair<Attribute,Double>> Attrs = new HashSet<Pair<Attribute,Double>>(Attributes);
        randomSplitTree(Root, Attrs);           
        
        boolean equilibrium = false;
        int iteration = 0;
    
        while(iteration < MaxIteration && !equilibrium){
        	double initialscore = 0;
        	double laterscore = 0;
        	
        	initialscore = gini.score(Root);
        	if(InnerNodes.toArray().length == 0)
                    return;
        	Node oldnode = (Node)InnerNodes.toArray()[Random.nextInt(InnerNodes.size())];    

        	Node newnode = new Node(oldnode);
        	
        	Pair<Attribute,Double> a;                      
        	HashSet<Pair<Attribute,Double>> subattrs = new HashSet<Pair<Attribute,Double>>(Attributes);    
        	deleteParAttr(oldnode,subattrs);
        	deleteChiAttr(oldnode,subattrs);            
        	if(subattrs.size()==0){
        		iteration++;
        		continue;
        	}
        	else{
               	a = (Pair<Attribute,Double>)subattrs.toArray()[Random.nextInt(subattrs.size())];
        	}
                
                double splitValue = a.getValue();
                Attribute splitAttr = a.getKey();
        	Instances[] Parts = partitionByAttr(newnode.data,splitAttr, splitValue);     
	     	Node[] Children = null;
                //new Node[a.numValues()];    
                
                 if(splitAttr.isNumeric())
                            Children = new Node[2];
                        else
                            Children = new Node[splitAttr.numValues()]; 
	     	
	    	    newnode.splitAttr = splitAttr;
		    newnode.children = Children;
                    newnode.splitValue = splitValue;
			
			for(int k=0; k < Parts.length; k++){
				Children[k] = new Node(Parts[k], newnode, k);
				
				makeLeafNode(Children[k]);
			}
		
			int minlen = (oldnode.children.length<newnode.children.length) ? oldnode.children.length : newnode.children.length;
			
			if(minlen==oldnode.children.length){          
				for(int l=0; l<minlen;l++){
					replaceNode(oldnode.children[l],newnode.children[l]);
			    }
				if(minlen!=newnode.children.length){
					for(int l=minlen; l<newnode.children.length; l++){
						HashSet<Pair<Attribute,Double>> subtreeAttrs = new HashSet<Pair<Attribute,Double>>(Attributes);
						randomSplitSubtree(newnode.children[l], subtreeAttrs);
					}
				}
			}
			else if(minlen==newnode.children.length){
				for(int l=0; l<minlen;l++){
					replaceNode(oldnode.children[l],newnode.children[l]);
			    }
			}
			
			if(oldnode.parent==null){
				newnode.parent = null;
				newnode.index = 0;
				Root = newnode;
			}
			else{
			    Node oldnodepar = oldnode.parent;
				newnode.parent = oldnodepar;
				newnode.index = oldnode.index;
				oldnodepar.children[oldnode.index] = newnode;
			}
		
			laterscore = gini.score(Root);
			
			double initialpro = expProbability(initialscore,splitmarkovbudget);
			double laterpro = expProbability(laterscore,splitmarkovbudget);
				
			double ratio = (double)(laterpro/initialpro);         
			if(ratio>=1){

				removeInnerNodes(oldnode);
				addInnerNodes(newnode);

				double totalscore = gini.score(Root);         
				equilibrium = isEquilibrium(totalscore);          
			}
			else{
				boolean replace = false;
				double randomdouble = Random.nextDouble();
				if(randomdouble<ratio){
					replace = true;
				}
				if(replace){

					removeInnerNodes(oldnode);
					addInnerNodes(newnode);

					double totalscore = gini.score(Root);          
					equilibrium = isEquilibrium(totalscore);          
				}
				else{

					if(oldnode.parent==null){
						Root = oldnode;
					}
					else{
						Node newnodepar = newnode.parent;
						oldnode.parent = newnodepar;
						oldnode.index = newnode.index;
						newnodepar.children[newnode.index] = oldnode;
					}
				}
			}

        	iteration++;                                 
        }

       // addNoise(Root, noisebudget);                                 
        
	}
        
        
        
        private void randomSplitTree(Node node, Set<Pair<Attribute,Double>> Attrs){

                Attrs = new HashSet<Pair<Attribute,Double>>(Attributes);

		if(node.depth >= MaxDepth){                     
			makeLeafNode(node);
			return;
		}

		for(int i=0; i< node.count.length; i++){        
			if(node.count[i] == node.data.numInstances()){
				makeLeafNode(node);
				return;
			}
		}
		
		deleteParAttr(node,Attrs);
		if(Attrs.size()==0){
			makeLeafNode(node);
			return;
		}
	
		makeInnerNode(node);
                InnerNodes.add(node);    

                Pair<Attribute,Double> SAInfo = (Pair<Attribute,Double>)Attrs.toArray()[Random.nextInt(Attrs.size())];
		Attribute splitAttr = SAInfo.getKey();//(Attribute)Attrs.toArray()[Random.nextInt(Attrs.size())];  
		double splitValue = SAInfo.getValue();
                
		Instances[] parts = partitionByAttr(node.data, splitAttr, splitValue);           
		Node[] children = null;
                
                        if(splitAttr.isNumeric())
                            children = new Node[2];
                        else
                            children = new Node[splitAttr.numValues()];        
		
		node.splitAttr = splitAttr; 
                node.splitValue = splitValue;
		node.children = children;
		
		for(int i=0; i < parts.length; i++){
			children[i] = new Node(parts[i], node, i);  
			randomSplitTree(children[i], Attrs);     
		}		
	}

	private void makeLeafNode(Node node){
		
		node.splitAttr = null;                                            
		node.children  = null;                                            
		node.isLeaf = true;	
	}
	
	private void makeInnerNode(Node node){

		node.isLeaf = false;	
	}

	private Instances[] partitionByAttr(Instances data, Attribute attr, double attrValue){
				
		Instances[] parts = null;
                
                if(attr.isNumeric())
                        parts = new Instances[2];
                 else       
                       parts =  new Instances[attr.numValues()];  
                
	    for(int i=0; i<parts.length; i++)
		{
			parts[i] = new Instances( data, data.numInstances() );         
		}
				
		Enumeration<Instance> instEnum = data.enumerateInstances(); 
		while(instEnum.hasMoreElements())                
		{
			Instance inst = instEnum.nextElement();
                        if(!attr.isNumeric()){
                            if(!inst.isMissing(attr))
                                  parts[(int)inst.value(attr)].add(inst);
                        }
                        else{
                            if(!inst.isMissing(attr)){
                                if(inst.value(attr)<=attrValue)
                                     parts[0].add(inst);
                                else parts[1].add(inst);
                            }
                        }
		}
				
		return parts;                                            
	}
	
	private void deleteParAttr(Node node, Set<Pair<Attribute,Double>> Attrs){
		
		while(node.parent != null){
                    Pair<Attribute, Double> a = new Pair(node.parent.splitAttr,node.parent.splitValue);
			Attrs.remove(a);
			node = node.parent;
		}
		return;
	}


        
        private void deleteChiAttr(Node node, Set<Pair<Attribute,Double>> Attrs){
	
		if(node.isLeaf==true)
			return;
		Attrs.remove(new Pair(node.splitAttr,node.splitValue));

		for(Node child: node.children){
				
			deleteChiAttr(child,Attrs);		
		}
	}

        
        private void replaceNode(Node oldnode, Node newnode){
		
		if(oldnode.isLeaf==true){
			makeLeafNode(newnode);
                        newnode.li = oldnode.li;
			return;
		}
		
		makeInnerNode(newnode);
		
		Attribute splitAttr = oldnode.splitAttr;
                double splitValue = oldnode.splitValue;
		
		newnode.splitAttr = splitAttr;
                newnode.splitValue = oldnode.splitValue;
		newnode.index = oldnode.index;
		
		Node[] children = null;
                
                    if(splitAttr.isNumeric())
                        children = new Node[2];
                    else children = new Node[splitAttr.numValues()];
                    
		Instances[] parts = partitionByAttr(newnode.data, splitAttr, splitValue);
		
		newnode.children = children;
	
		for(int i=0;i<parts.length;i++){
			children[i] = new Node(parts[i],newnode,i);
			
			replaceNode(oldnode.children[i],children[i]);
		}
	}
	
	
	private void removeInnerNodes(Node node){
		
		if(node.isLeaf==true)
			return;
		for(Node child: node.children){
			removeInnerNodes(child);
		}
		InnerNodes.remove(node);
	}

	private void addInnerNodes(Node node){
		
		if(node.isLeaf==true)
			return;
		for(Node child: node.children){
			addInnerNodes(child);
		}
		InnerNodes.add(node);
	}

	private void randomSplitSubtree(Node node, HashSet<Pair<Attribute,Double>> Attrs){
		
		Attrs = new HashSet<Pair<Attribute,Double>>(Attributes);

		if(node.depth >= MaxDepth){                     
			makeLeafNode(node);
			return;
		}
		if(node.data.numInstances()==0){
			makeLeafNode(node);
			return;
		}
		for(int i=0; i< node.count.length; i++){        
			if(node.count[i] == node.data.numInstances()){
				makeLeafNode(node);
				return;
			}
		}
		
		deleteParAttr(node,Attrs);
		if(Attrs.size()==0){
			makeLeafNode(node);
			return;
		}
	
		makeInnerNode(node);
		
		Pair<Attribute,Double> splitAttr = (Pair<Attribute,Double>)Attrs.toArray()[Random.nextInt(Attrs.size())];  
		 double splitValue = splitAttr.getValue();
                Attribute splitA = splitAttr.getKey();
		Instances[] parts = partitionByAttr(node.data, splitA, splitValue);           
		
                Node[] children = null;//new Node[splitAttr.numValues()];        
		
                if(splitA.isNumeric())
                            children = new Node[2];
                        else
                            children = new Node[splitA.numValues()];        
		
		node.splitAttr = splitA; 
                node.splitValue = splitValue;
		node.children = children;
		
		for(int i=0; i < parts.length; i++){
			children[i] = new Node(parts[i], node, i);  
			randomSplitSubtree(children[i], Attrs);     
		}
	}
	
    private void addNoise(Node node, BigDecimal budget){          

    	if(node.isLeaf == true)                                  
    	{ 
    		addNoiseDistribution(node.count, budget);   
    		node.updateDist();                                
	    	return;
    	}

    	for(Node child : node.children)                    
        {
        	addNoise(child, budget);
        }
    }
   
    private void addNoiseDistribution(double[] count, BigDecimal budget){
    	                                       
    	int maxIndex = Utils.maxIndex(count);                  
    	
    	for(int i=0; i<count.length; i++)
    	{
    		count[i] += laplace(BigDecimal.ONE.divide(budget, MATH_CONTEXT));      
 
    		if(count[i] < 0)	                      
    			count[i] = 0;
    	}
 
    	double sum = Utils.sum(count);                           
    	if(sum <= 0){	                         
    		count[maxIndex] = 1;           
    	}
    } 

    private double laplace(BigDecimal bigBeta){                
    	
    	double miu = 0.;                                             
    	double beta=bigBeta.doubleValue();                
    	
        double uniform = Random.nextDouble()-0.5;            
        return miu-beta*((uniform>0) ? -Math.log(1.-2*uniform) : Math.log(1.+2*uniform));  
        
    }

  	private double expProbability(double score, BigDecimal epsilon)        
  	{
  		return Math.exp(epsilon.doubleValue() * score / (2 * GiniSen));
  	}                  
  	
  	final int BufferSize = 500;   
  	double[] ScoreBuffer = new double[BufferSize];

  	int InitPointer = 0;           
  	int Pointer = 0;               
  	double variance = -1;                 
  	
  	private boolean isEquilibrium(double newScore)
  	{
  		if(InitPointer == BufferSize){
  			variance = Utils.variance(ScoreBuffer);
  			if(variance < EquilibriumThreshold){
  				return true;
  			}
  			InitPointer++;
  		}
  		if( InitPointer < BufferSize ){                 
  			ScoreBuffer[InitPointer++] = newScore;       
  			
  			return false;                                    
  		}
  		if(Pointer==BufferSize)
  			Pointer = 0;
  		ScoreBuffer[Pointer++] = newScore;                                 
  		
  		variance = Utils.variance(ScoreBuffer);          

  		if(variance < EquilibriumThreshold){         
  			InitPointer = 0;
  			Pointer = 0;                                
  			return true; 
  		}
  		return false;                                   
  	}
  	
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException { 	
    	
    	assert( instance.hasMissingValue() == false);
    	
    	Node node = Root;                                   
    	while(node.isLeaf == false){                          
    		Attribute attr = node.splitAttr;
    		node = node.children[ (int)instance.value(attr) ];
    	}
    	
    	return Utils.maxIndex(node.dist);          
    }

	public double[] distributionForInstance(Instance instance) throws NoSupportForMissingValuesException {	
		
		assert( instance.hasMissingValue() == false);
		
		Node node = Root;                                    
    	while(node.isLeaf == false){                     
    		Attribute attr = node.splitAttr;
    		node = node.children[ (int)instance.value(attr) ];
    	}
    	
    	return node.dist;                           
	}                                       
	
    public String toString() {                    

        return toString(Root);
       
    }

	protected String toString(Node node) {
		
		int level = node.depth;

		StringBuffer text = new StringBuffer();

		if (node.isLeaf == true) {                              

			text.append("  [" + node.data.numInstances() + "]");
			text.append(": ").append(
					node.data.classAttribute().value((int) Utils.maxIndex(node.dist)));
			
			text.append("   Counts  " + distributionToString(node.dist));
			
		} else {                                                
			
			text.append("  [" + node.data.numInstances() + "]");
			for (int j = 0; j < node.children.length; j++) {
				
				text.append("\n");
				for (int i = 0; i < level; i++) {
					text.append("|  ");
				}
				
				text.append(node.splitAttr.name())
					.append(" = ")
					.append(node.splitAttr.value(j));
				
				text.append(toString(node.children[j]));
			}
		}
		return text.toString();
	}

    private String distributionToString(double[] distribution)
    {
           StringBuffer text = new StringBuffer();
           text.append("[");
           for (double d:distribution)
                  text.append(String.format("%.2f", d) + "; ");
           text.append("]");
           return text.toString();             
    }

}
