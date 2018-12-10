#https://www.youtube.com/watch?time_continue=1&v=WOXsdISQjlM

##jika tensorflow tidak dikenali oleh R pada saat menjalankan function to_categorical
#library(reticulate)
#conda_remove("r-tensorflow")
#library(keras)
#install_keras()

##Refresh R-Studio
#library(keras)
#use_condaenv("r-tensorflow")
#dataset_mnist()

library(keras)
library(tensorflow)
library(EBImage)
setwd("F://KULIAH//SEMESTER 7//Trending Topic Statistics//Image Classification Grab Vs Gojek//Dataset/") #set working directory
save_in=("F://KULIAH//SEMESTER 7//Trending Topic Statistics//Image Classification Grab Vs Gojek//Resized/")

gambar=list.files()
w=250
h=250
for(i in 1:length(gambar))
{result=tryCatch({
  imgname=gambar[i]
  img=readImage(imgname)
  img_resized=resize(img,w=w,h=h)
  path=paste(save_in, imgname, sep="")
  writeImage(img_resized,path,quality = 90)
  print(paste0("done",i,sep=""))
},
error=function(e){print(e)})}

#mulai analisis dengan gambar baru yg sudah di resize
setwd("F://KULIAH//SEMESTER 7//Trending Topic Statistics//Image Classification Grab Vs Gojek//Resized/")
gambar_resized=list.files()
gambar_resized

gambar_resized=lapply(gambar_resized,readImage)
str(gambar_resized)
display(gambar_resized[[1]])

train=gambar_resized[c(1:40,51:90)]
test=gambar_resized[c(41:50,91:100)]

for (i in 1:80){train[[i]]=resize(train[[i]],32,32)}
for (i in 1:20){test[[i]]=resize(test[[i]],32,32)}

train=combine(train)
x=tile(train,80)
plot.new()
display(x,title("Train"))
dim(train)

test=combine(test)
y=tile(test,20)
display(y, title("Test"))
dim(test)

train=aperm(train,c(4,1,2,3)) #menyusun dimensi(dim) sesuai dengan ketentuan CNN
test=aperm(test,c(4,1,2,3))
dim(train)

#mulai klasifikasi
trainy=c(rep(1,40),rep(2,40))
testy=c(rep(1,10),rep(2,10))

#membuat label data menjadi kategorik
trainLabels=to_categorical(trainy)
testLabels=to_categorical(testy)


#Video 2: https://www.youtube.com/watch?v=sFGsshgWNg0
model=keras_model_sequential()
model%>%
  layer_conv_2d(filters=32,
                kernel_size = c(3,3),
                activation = "relu",
                input_shape = c(32,32,3))%>% #relu=rectivied linear unit. mengambil nilai tertinggi untuk aktivasi
  layer_conv_2d(filters = 32,
                kernel_size = c(3,3),
                activation = "relu")%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>%
  layer_dropout(rate=0.01) %>% #untuk regularisasi pada node yang punya nilai kecil tidak akan diteruskan/dikerjakan
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = "relu")%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>%
  layer_dropout(rate=0.01) %>%
  layer_flatten()%>% #merubah menjadi vektor
  layer_dense(units=256, activation = "relu")%>%
  layer_dropout(rate=0.01) %>%
  layer_dense(units=3,activation = "softmax")%>% #unit=banyak objek yg akan di klasifikasi, softmax=hampir mirip dgn relu
  compile(loss="categorical_crossentropy",
          optimizer = optimizer_sgd(lr=0.01,
                                    decay = 1e-06,
                                    momentum = 0.9,
                                    nesterov=T),
          metrics=c('accuracy'))#sgd=stokastik gladiand descend (secara stokastik random memilih mana yg dengan cepat menuju global optimum)
summary(model)

proses=model%>%
  fit(train,
      trainLabels,
      epoch=50,
      batch_size = 32,
      validation_split = 0.2)
plot(proses)

model%>%evaluate(train, trainLabels)
pred=model%>%predict_classes(train)
pred
table(predicted=pred, actual=trainy)
prob=model%>%predict_proba(train)
cbind(prob,predict_classes=pred,actual=trainy)

model%>%evaluate(test,testLabels)
pred=model%>%predict_classes(test)
pred
table(predicted=pred, actual=testy)
prob=model%>%predict_proba(test)
cbind(prob,predict_classes=pred,actual=testy)

#(32,32,)
#(30,30,32) --> 30 dapat dari perpindahan convo 3x3 (berdasarkan padding dan strech)
#(28,28,32)
#maxpool (2) digunakan untuk mereduksi jadi setengahya = 14, untuk mencari nilai pixel terbesarnya 
#aktivasi fungsi, y=(1/1+e^x) --> diturunkan dy/dx=f(x)(1-f(x))
#relu=rectified linear unit (menggunakan nilai terbesar pada neuron)