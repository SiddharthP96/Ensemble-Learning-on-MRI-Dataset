from numpy import *
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
import SimpleITK as sitk

def makedata():
#Define a function that converts .mha to numpy arrays and *normalises the data 
	no=6082560
	ne=no*20
	(a,b,c)=(176,216,160)
	X=zeros((ne,4))		#T1,T1c,T2,Flair
	Y=zeros((ne,1))		#for hgg belongs to {0,1,2,3,4}
	X6=zeros((ne,12))   #mean in 2D plane 
	X7=zeros((ne,8))	#context based
	co=0
	for f in range(20):
		print("Making data for image number %i" %(f+1))
		m4=r'C:\Users\sid\Desktop\Project\BRATS_2013\BRATS-2\Image_Data\HG\ALL\Flair\F (%i).mha' %(f+1)
		m3=r'C:\Users\sid\Desktop\Project\BRATS_2013\BRATS-2\Image_Data\HG\ALL\T2\T2 (%i).mha' %(f+1)
		m2=r'C:\Users\sid\Desktop\Project\BRATS_2013\BRATS-2\Image_Data\HG\ALL\T1c\T1c (%i).mha' %(f+1)
		m1=r'C:\Users\sid\Desktop\Project\BRATS_2013\BRATS-2\Image_Data\HG\ALL\T1\T1 (%i).mha' %(f+1)
		n=r'C:\Users\sid\Desktop\Project\BRATS_2013\BRATS-2\Image_Data\HG\ALL\OT\OT (%i).mha' %(f+1)
#X=[T1:T1c:T2:Flair] Y=[OT]	
		m4no=sitk.ReadImage(m4)
		m3no=sitk.ReadImage(m3)
		m2no=sitk.ReadImage(m2)
		m1no=sitk.ReadImage(m1)
		nno=sitk.ReadImage(n)
#SimpleITK stuff
		m4n=sitk.GetArrayFromImage(m4no)
		m3n=sitk.GetArrayFromImage(m3no)
		m2n=sitk.GetArrayFromImage(m2no)
		m1n=sitk.GetArrayFromImage(m1no)
		nn=sitk.GetArrayFromImage(nno)
#Initialising count for every new image		
		count=0+f*no	
		for i in range(a):
			for j in range(b):
					for k in range(c):
						if (i>9 and i<a-9 and j>9 and j<b-9 and k>9 and k<c-9):
							size=1
							a1=mean(m1n[i-size:i+size,j-size:j+size,k])
							a2=mean(m2n[i-size:i+size,j-size:j+size,k])
							a3=mean(m3n[i-size:i+size,j-size:j+size,k])
							a4=mean(m4n[i-size:i+size,j-size:j+size,k])
							'''a5=mean(m1n[i,j-size:j+size,k-size:k+size])
							a6=mean(m2n[i,j-size:j+size,k-size:k+size])
							a7=mean(m3n[i,j-size:j+size,k-size:k+size])
							a8=mean(m4n[i,j-size:j+size,k-size:k+size])
							a9=mean(m1n[i-size:i+size,j,k-size:k+size])
							a10=mean(m2n[i-size:i+size,j,k-size:k+size])
							a11=mean(m3n[i-size:i+size,j,k-size:k+size])
							a12=mean(m4n[i-size:i+size,j,k-size:k+size])'''
							size=4
							b1=mean(m1n[i-size:i+size,j-size:j+size,k])
							b2=mean(m2n[i-size:i+size,j-size:j+size,k])
							b3=mean(m3n[i-size:i+size,j-size:j+size,k])
							b4=mean(m4n[i-size:i+size,j-size:j+size,k])
							'''b5=mean(m1n[i,j-size:j+size,k-size:k+size])
							
							
							
							
							
							
							
							
							
							
							b6=mean(m2n[i,j-size:j+size,k-size:k+size])
							b7=mean(m3n[i,j-size:j+size,k-size:k+size])
							b8=mean(m4n[i,j-size:j+size,k-size:k+size])
							b9=mean(m1n[i-size:i+size,j,k-size:k+size])
							b10=mean(m2n[i-size:i+size,j,k-size:k+size])
							b11=mean(m3n[i-size:i+size,j,k-size:k+size])
							b12=mean(m4n[i-size:i+size,j,k-size:k+size])'''
							size=9
							c1=mean(m1n[i-size:i+size,j-size:j+size,k])
							c2=mean(m2n[i-size:i+size,j-size:j+size,k])
							c3=mean(m3n[i-size:i+size,j-size:j+size,k])
							c4=mean(m4n[i-size:i+size,j-size:j+size,k])
							'''c5=mean(m1n[i,j-size:j+size,k-size:k+size])
							c6=mean(m2n[i,j-size:j+size,k-size:k+size])
							c7=mean(m3n[i,j-size:j+size,k-size:k+size])
							c8=mean(m4n[i,j-size:j+size,k-size:k+size])
							c9=mean(m1n[i-size:i+size,j,k-size:k+size])
							c10=mean(m2n[i-size:i+size,j,k-size:k+size])
							c11=mean(m3n[i-size:i+size,j,k-size:k+size])
							c12=mean(m4n[i-size:i+size,j,k-size:k+size])'''
							X6[count,:]=array([a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4])
#Adding them context based features
#Need to find averages at points (a-4,b,c) (a+4,b,c) (a,b-4,c) (a,b+4,c) (a,b,c+4) (a,b,c-4)
						X[count,:]=array([m1n[i,j,k],m2n[i,j,k],m3n[i,j,k],m4n[i,j,k]])
						Y[count]=array([nn[i,j,k]])
						count+=1
#for a given coordinate (a,b,c) count(a,b,c)=f*6082560+a*176+b*216+c*160
	X1a=exp((4/1024)*X[:,:2]+2)
	X1b=exp((5/1024)*X[:,2:]+3)
	X2a=log((1/1024)*X[:,0]+1)
	X2b=log((5/1024)*X[:,1]+1)
	X2c=log((6/1024)*X[:,2]+1)
	X2d=log((2/1024)*X[:,3]+1)
	X3=X[2,:]-X[1,:]
	X4=X[3,:]-X[2,:]
	X5=X[3,:]-X[1,:]
	co=f*6082560
	for f in range (20):
		for i in range(a):
				for j in range(b):
						for k in range(c):
							if (i>9 and i<a-9 and j>9 and j<b-9 and k>9 and k<c-9):
								a1=argmax(array([X6[f*no+(i-4)*a+j*b+k*c,1],X6[f*no+(i+4)*a+j*b+k*c,1],X6[f*no+i*a+(j-4)*b+k*c,1],X6[f*no+i*a+(j+4)*b+k*c,1],X6[f*no+i*a+j*b+(k+4)*c,1],X6[f*no+i*a+j*b+(k-4)*c,1]]))
								a2=argmin(array([X6[f*no+(i-4)*a+j*b+k*c,1],X6[f*no+(i+4)*a+j*b+k*c,1],X6[f*no+i*a+(j-4)*b+k*c,1],X6[f*no+i*a+(j+4)*b+k*c,1],X6[f*no+i*a+j*b+(k+4)*c,1],X6[f*no+i*a+j*b+(k-4)*c,1]]))
								a3=argmax(array([X6[f*no+(i-4)*a+j*b+k*c,2],X6[f*no+(i+4)*a+j*b+k*c,2],X6[f*no+i*a+(j-4)*b+k*c,2],X6[f*no+i*a+(j+4)*b+k*c,2],X6[f*no+i*a+j*b+(k+4)*c,2],X6[f*no+i*a+j*b+(k-4)*c,2]]))
								a4=argmin(array([X6[f*no+(i-4)*a+j*b+k*c,2],X6[f*no+(i+4)*a+j*b+k*c,2],X6[f*no+i*a+(j-4)*b+k*c,2],X6[f*no+i*a+(j+4)*b+k*c,2],X6[f*no+i*a+j*b+(k+4)*c,2],X6[f*no+i*a+j*b+(k-4)*c,2]]))
								a5=argmax(array([X6[f*no+(i-4)*a+j*b+k*c,3],X6[f*no+(i+4)*a+j*b+k*c,3],X6[f*no+i*a+(j-4)*b+k*c,3],X6[f*no+i*a+(j+4)*b+k*c,3],X6[f*no+i*a+j*b+(k+4)*c,3],X6[f*no+i*a+j*b+(k-4)*c,3]]))
								a6=argmin(array([X6[f*no+(i-4)*a+j*b+k*c,3],X6[f*no+(i+4)*a+j*b+k*c,3],X6[f*no+i*a+(j-4)*b+k*c,3],X6[f*no+i*a+(j+4)*b+k*c,3],X6[f*no+i*a+j*b+(k+4)*c,3],X6[f*no+i*a+j*b+(k-4)*c,3]]))
								a7=argmax(array([X6[f*no+(i-4)*a+j*b+k*c,4],X6[f*no+(i+4)*a+j*b+k*c,4],X6[f*no+i*a+(j-4)*b+k*c,4],X6[f*no+i*a+(j+4)*b+k*c,4],X6[f*no+i*a+j*b+(k+4)*c,4],X6[f*no+i*a+j*b+(k-4)*c,4]]))
								a8=argmin(array([X6[f*no+(i-4)*a+j*b+k*c,4],X6[f*no+(i+4)*a+j*b+k*c,4],X6[f*no+i*a+(j-4)*b+k*c,4],X6[f*no+i*a+(j+4)*b+k*c,4],X6[f*no+i*a+j*b+(k+4)*c,4],X6[f*no+i*a+j*b+(k-4)*c,4]]))
								X7[co,:]=array([a1,a2,a3,a4,a5,a6,a7,a8])
							co=+1	
					
	X=concatenate((X,X1a),1)
	X=concatenate((X,X1b),1)
	X=concatenate((X,X2a),1)
	X=concatenate((X,X2b),1)
	X=concatenate((X,X2c),1)
	X=concatenate((X,X2d),1)
	X=concatenate((X,X3),1)
	X=concatenate(X,X4),1)
	X=concatenate((X,X5),1)
	X=concatenate((X,X6),1)
	X=concatenate((X,X7),1)
	return X,Y
	
#X=[t1,t1c,t2,fl,log(four),exp(four),t2-t1c,fl-t2,fl-t1c,mean of 5x5 plane in xy plane(four),mean of 5x5 plane in yz plane(four)
#mean of 5x5 plane in xz plane(four), context based(two)]
(X,Y)=makedata()
clf=ExtraTreesClassifier()
clf.fit(X,Y)

Yt=clf.predict(X)
tp=0
fn=0
fp=0
for i in range(size(Y)):
	for k in range(3):
		if (Y[i]==k+1 and Yt[i]==k+1):
			tp+=1
		elif (Y[i]!=k+1 and Yt[i]==k+1):
			fp+=1
		elif (Y[i]==k+1 and Yt[i]!=k+1):
			fn+=1

dsc=2*tp/(fp+fn+2*tp) #Dice Similarity Coefficient 
ppv=tp/(tp+fp)		  #Posotive Predictive Value	
s=tp/(tp+fn)		  #Senitivity

print(dsc,ppv,s)
