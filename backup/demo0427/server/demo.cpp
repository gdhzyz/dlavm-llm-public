#include "./server.h"
#include "./xdma_lib/xdma_public.h"
#include "./rw_cmd/chatglm_dynamic_control_0427_1717.h"

Module RPCClient::mod = chatglm_dynamic_control_0427_1717;

int __cdecl main()
{
    //****************************** open device ****************************// 
    HANDLE user_device;
    HANDLE bypass_device;
    HANDLE c2hx_device[NUM_OF_RW_CH];
    HANDLE h2cx_device[NUM_OF_RW_CH];   
    open_device(&user_device, &bypass_device, &c2hx_device[0], &h2cx_device[0]);
    //****************************** open device ****************************//

    read_bin_split("./embedding", 16);
    init_Socket();
    int sockfd;
    if ((sockfd = socket(AF_INET,SOCK_STREAM,0)) == -1) { 
        perror("socket"); 
        return -1; 
    }
    
    struct sockaddr_in servaddr;
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(8123); // server's port
    servaddr.sin_addr.s_addr=inet_addr("10.20.72.156");//server's ip
    if (bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) == -1) { 
        perror("bind"); 
        return -1; 
    }

    if (listen(sockfd, 10) == -1) {
        perror("listen");
        return -1;
    }
    std::cout << "FPGA ChatGLM2 Server Waiting..." << std::endl;

    struct sockaddr_in client_addr;
    int addr_len = sizeof(struct sockaddr_in);
    int newsock;
    while (1) {
        if ((newsock = accept(sockfd, (struct sockaddr*)&client_addr, &addr_len)) == INVALID_SOCKET) {
            std::cout << "Fail" << WSAGetLastError() << std::endl;
            return 0;
        } else {
            std::cout << "[info] Connect " << inet_ntoa(client_addr.sin_addr) << std::endl;
        }

        RPCClient client(newsock, user_device, c2hx_device[0], h2cx_device[0], data_in, data_out);
        while (client.main_loop()) {};
        std::cout << "[info] Disconnect " << inet_ntoa(client_addr.sin_addr) << std::endl;
    }

    return 0;
}