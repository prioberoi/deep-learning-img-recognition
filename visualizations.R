library(stringr)
library(ggplot2)
library(Rmisc)

#############################
# learning rate plot of "inv"
############################# we may want to have all the different learning rates on here
base_lr <- 0.01
gamma <- 0.0001
power <- 0.75
lr <- c()
for(iter in 1:100){
  lr <- c(lr, (base_lr * (1 + gamma * iter) ^ (- power)))
}

lr <- data.frame(iter = c(1:100), inv = lr)

ggplot(lr, aes(x = iter, y = inv)) +
  geom_line()

#############################
# visualizing training and test output from model
#############################
# parameter setup in solver
model_name <- "model_1"
max_iter <- 10000
display <- 100
test_interval <- 500
iter <- seq(from = 0, to = max_iter, by = display)
train_size <-  60000
batch_size <-  64
iterations_per_epoch <- train_size/batch_size
total_epochs <- ceiling(min((train_size/iterations_per_epoch), (max_iter/iterations_per_epoch)))
epochs <- sort(rep(0:total_epochs, length.out = length(iter)), decreasing = FALSE)

# get console output
output <- read.csv(paste0("/Users/prioberoi/Documents/pri_reusable_code/deep-learning-img-recognition/console_output_", model_name), header = FALSE, sep = "\n")
output$V1 <- as.character(output$V1)

# plot of learning rate
lr <- output[grep("Iteration [0-9]+, lr = ", output$V1),]
lr <- str_extract(lr, "lr = [0-9]+\\.[0-9]+")
lr <- str_extract(lr, "[0-9]+\\.[0-9]+")
lr <- as.numeric(lr)
lr <- data.frame(iter = seq(from = 0, to = (max_iter-display), by = 100), lr = lr)

############################# 
# training loss 
#############################
train_loss <- output[grep("Iteration [0-9]+, loss =", output$V1),]
train_loss <- str_extract(train_loss, "loss = [0-9]+\\.[0-9]+")
train_loss <- str_extract(train_loss, "[0-9]+\\.[0-9]+")
train_loss <- as.numeric(train_loss)

model <- data.frame(iter = iter,
                    epoch = epochs,
                    train_loss = train_loss, 
                    test_loss = rep(NA, length(iter)),
                    test_accuracy = rep(NA, length(iter)))

############################# 
# validation loss
#############################
test_loss <- output[grep("Test net output #1: loss = ", output$V1),]
test_loss <- str_extract(test_loss, "loss = [0-9]+\\.[0-9]+")
test_loss <- str_extract(test_loss, "[0-9]+\\.[0-9]+")
test_loss <- as.numeric(test_loss)

############################# 
# validation accuracy
#############################
test_accuracy <- output[grep("Test net output #0: accuracy = ", output$V1),]
test_accuracy <- str_extract(test_accuracy, "accuracy = [0-9]+\\.[0-9]+")
test_accuracy <- str_extract(test_accuracy, "[0-9]+\\.[0-9]+")
test_accuracy <- as.numeric(test_accuracy)

test <- data.frame(iter = seq(from = 0, to = max_iter, by = test_interval), test_loss = test_loss, test_accuracy = test_accuracy)

model[model$iter %in% test$iter, c("test_loss", "test_accuracy")] <- test[,c("test_loss", "test_accuracy")]

#############################
# plot
#############################

a <- ggplot(lr, aes(x = iter, y= lr)) +
  geom_line(group = 1) +
  ggtitle(paste0("Learning Rate per Iteration ", model_name))

b <- ggplot(model) +
  geom_line(data = model, aes(x = iter, y = train_loss, colour = "train_loss"), group = 1) +
  geom_line(data = model[!is.na(model$test_loss),], aes(x = iter, y = test_loss, colour = "test_loss"), group = 1) +
  ggtitle(paste0("Training and Test Loss per Iteration ", model_name)) +
  labs(y = "loss") +
  geom_vline(xintercept = iterations_per_epoch, linetype = 2) +
  annotate("text", x = iterations_per_epoch, y = 2, size = 3, hjust = 0,
           label = paste0("First epoch"))

c <- ggplot(model) +
  geom_line(data = model[!is.na(model$test_accuracy),], aes(x = iter, y = test_accuracy, colour = "test_accuracy"), group = 1) +
  ggtitle(paste0("Test Accuracy per Iteration ", model_name)) +
  geom_vline(xintercept = iterations_per_epoch, linetype = 2) +
  annotate("text", x = iterations_per_epoch, y = 2, size = 3, hjust = 0,
           label = paste0("First epoch"))
  
multiplot(a,b,c, cols = 1)
